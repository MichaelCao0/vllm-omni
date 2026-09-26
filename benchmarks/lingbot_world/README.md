# LingBot-World Benchmarks — Realtime Streaming Cadence

`benchmark_lingbot_world_realtime.py` measures LingBot-World 2.0 as it is
actually deployed: one `WS /v1/realtime/video` session that streams one video
chunk per AR block, for as long as the world keeps running.

## Why this is not the diffusion serving benchmark

[`benchmarks/diffusion`](../diffusion/README.md) measures "one request, one
finished video" and reports request throughput. LingBot-World has neither shape:

- **One request is the whole rollout.** `prepare_encode` runs once, then every AR
  block is four DMD steps plus one clean-KV commit forward, and each block
  streams out as a chunk. Throughput in requests per second is meaningless.
- **There is exactly one session.** The stepwise deploy config pins
  `max_num_seqs: 1`, and the AR-Diffusion consumer rejects a stage with more than
  one replica, because session-affine routing does not exist yet. There is no
  concurrency to sweep.
- **What matters is cadence.** A deployment is healthy when chunk *N + 1* arrives
  before a viewer finishes watching chunk *N*. That is a latency distribution
  against a wall clock, not a throughput number.

## Quick start

Serve the model, then point the benchmark at it. The benchmark never starts a
server.

```bash
# Terminal 1 — one GPU, eager (portable smoke configuration)
vllm serve robbyant/lingbot-world-v2-14b-causal-fast-diffusers \
  --omni \
  --deploy-config benchmarks/lingbot_world/configs/single_gpu_eager.yaml \
  --port 8000

# Terminal 2
pip install websockets
python benchmarks/lingbot_world/benchmark_lingbot_world_realtime.py \
  --host 127.0.0.1 --port 8000 \
  --num-chunks 16 \
  --output-json /tmp/lingbot_world_realtime.json
```

With no `--image`, the first frame comes from the shared vLLM image asset, so the
default workload needs no setup and this repository carries no benchmark binary.
Pass `--image /path/to/frame.png` to condition the rollout on your own frame.

### Realtime topology

The single-GPU config is a smoke configuration, not a realtime one: one card runs
all five forwards of every block serially and nothing is compiled. Use the
four-GPU config for numbers that mean anything about realtime behavior.

```bash
vllm serve robbyant/lingbot-world-v2-14b-causal-fast-diffusers \
  --omni \
  --deploy-config benchmarks/lingbot_world/configs/usp4_compiled.yaml \
  --port 8000

python benchmarks/lingbot_world/benchmark_lingbot_world_realtime.py \
  --port 8000 --num-chunks 40 --warmup-sessions 1 --sessions 3 \
  --target-fps 12 --output-json /tmp/lingbot_world_repeated.json
```

`usp4_compiled.yaml` sets pure Ulysses sequence parallelism over four ranks and
`enforce_eager: false`. The pipeline validates the parallel shape itself:
`sequence_parallel_size` must equal `ulysses_degree`, `ring_degree` and
`allgather_degree` must be 1, `ulysses_mode` must be `strict`, and pipeline,
CFG, and VAE parallel sizes above 1 are rejected. This benchmark config sets
`tensor_parallel_size: 1`; the pipeline validator does not reject larger TP sizes.

Compiled mode can pay compilation and capture costs on the first rollout.
Use `--warmup-sessions 1` before measured repetitions. Excluding the first six
chunks addresses the attention-window ramp; it does not guarantee that
compilation and capture have finished.

## Metrics

| Metric | Meaning |
| --- | --- |
| `ttfc_ms` | **Time to first chunk**: `session.start` sent until the first chunk's bytes arrive. Pays for prompt encode, first-frame VAE encode, and any compile or capture. |
| `interval_all` | Inter-arrival latency across every chunk: mean, median, std, min, p90, p95, p99, max. |
| `interval_steady` | The same summary over steady-state chunks only. |
| `video_rtf` | Wall seconds spent per second of video, matching vLLM-Omni's own RTF (`generation_s / video_duration`). **Lower is better; below 1.0 is real time.** |
| `video_rtfx` | The reciprocal, under the name this repo uses for it: video seconds per wall second, where higher is better. |
| `steady_video_rtf` | `video_rtf` restricted to steady-state chunks. |
| `chunk_deadline_ms` | Wall-clock budget one steady chunk must meet to hold real time: 12 frames ÷ `--target-fps`. Override directly with `--chunk-slo-ms`. |
| `slo_attainment` | Fraction of steady chunks that met the deadline. |
| `playback` | Underrun count, total stall, and stall ratio from a simulated viewer. |

### Which frame rate counts as real time

**The checkpoint declares no frame rate.** There is none in `model_index.json`, the
scheduler config, or the transformer config, and the pipeline never reads one. A
chunk is twelve pixel frames; how long those frames *last* is a property of how the
output is played, not of the model.

That choice moves the verdict without moving anything measured:

| basis | video per chunk | deadline | RTF at a measured 1357 ms/chunk |
| --- | --- | --- | --- |
| 16 fps | 0.750 s | 750 ms | 1.809 |
| 12 fps | 1.000 s | 1000 ms | 1.357 |

So `--fps` (sent to the server, which uses it to label the muxed MP4) and
`--target-fps` (the real-time basis for RTF, the deadline, and the playback
simulation) are separate flags. `--target-fps` defaults to `--fps`; set it when the
two differ, and the report prints both. State the basis whenever you quote a number
— an RTF without its fps is not a measurement. (RTF is a cost here, so a bigger number is a
worse result; a steady chunk's RTF is simply its interval divided by its video duration.)
An RTF above 1 means the measured deployment is slower than real time at the
stated target frame rate. The benchmark measures this failure as well as success.

### Why warmup chunks are not optional

The DiT attends over a sliding window of `sliding_window_num_frames = 18` latent
frames with `sink_size = 9`, so the key count a block attends over keeps growing
until eighteen latent frames of history exist — six chunks at three frames per
block. **Chunk cost is not stationary before chunk six.** A mean over a short
rollout is therefore lower than anything a running session will sustain, which is
why `--warmup-chunks` defaults to 6 and steady-state metrics are reported
separately. Keep `--num-chunks` comfortably above the warmup, or the benchmark
tells you no chunk reached steady state. The terminal chunk skips next-chunk
preparation, so it is excluded from steady metrics too; it remains in overall
RTF, all-chunk intervals, and playback simulation.

The default 16-chunk rollout provides only nine steady intervals. Percentiles
are interpolated sample summaries; the report warns when fewer than 100
intervals contribute, and even 100 samples do not guarantee a stable p99.
Use longer rollouts for tail analysis and at least `--sessions 3` after
`--warmup-sessions 1` for reported comparisons. Session warmup pays compilation
cost; chunk warmup separately excludes the attention-window ramp. The command
above yields 33 steady intervals per session (indices 6 through 38), or 99 across
three sessions. Report each session's mean and the spread across sessions; this
sample still does not establish a reliable p99.

### Why playback is simulated

A p99 interval alone cannot say whether a viewer saw a stall: the same p99 is
invisible behind a three-chunk buffer and a visible freeze behind none. The
benchmark replays chunk arrivals against a wall clock — playback starts once
`--playback-buffer-chunks` chunks have landed and then consumes video at `--target-fps`
— and reports every moment the player ran dry.

### What the client-side number is worth

On a four-GPU Ulysses run the client's mean inter-arrival over all fifteen
post-first-chunk intervals was **1326.743 ms**, against the server's own
`StageRequestStats.inter_output_latency_ms` of **1327.479 ms** for the same
request — a difference of **0.74 ms**. This cross-check shows agreement in
average output cadence for that run, including warmup and the terminal chunk.
It does not measure encoding or transport latency: a fixed delivery delay can
cancel out of inter-arrival differences. Attributing that latency requires
per-chunk timestamps at both boundaries or server-side profiling.

Compare matching interval populations when investigating discrepancies; both
measurement bugs and delivery buffering can change the observed cadence.

### What this benchmark cannot see

Chunk metadata carries no server-side timestamps, so the benchmark cannot split
DiT time from VAE decode time; use `--enable-diffusion-pipeline-profiler` on the
server for that. It also reports no GPU memory, which is a server-side quantity.

## Custom rollouts

`--workload rollout.json` replaces the built-in workload. The default
`camera_mode: "script"` generates a fixed camera script from `num_chunks` and
`camera_pattern`. Supply `camera_action_script` directly to control every
frame's action; it takes precedence over those two fields and must contain one
three-entry action list per chunk.

For commands sent during generation, use `camera_mode: "live"` and an explicit
`num_chunks`. The server holds the camera still until the first update. Live
commands and a fixed `camera_action_script` or `camera_pattern` are mutually
exclusive. LingBot-World currently supports camera interaction only;
`prompt_updates` are rejected locally because the pipeline has no prompt handler.

```json
{
  "prompt": "The camera moves slowly forward through the scene.",
  "image": "/path/to/first_frame.png",
  "num_chunks": 24,
  "camera_mode": "live",
  "fps": 16,
  "seed": 42,
  "camera_updates": [
    {
      "after_chunk": 2,
      "mode": "target",
      "translation": [0.0, 0.0, 1.0],
      "rotation": [0.0, 0.0, 0.0, 1.0],
      "transition_chunks": 2
    },
    {
      "after_chunk": 8,
      "mode": "velocity",
      "translation": [0.01, 0.0, 0.0]
    }
  ]
}
```

`after_chunk` is zero-based and triggers after that chunk's media bytes arrive.
It must precede the final chunk and be unique within the workload; event IDs
are generated as `camera-after-N`. This is a chunk-triggered workload, not an
independent real-user arrival process. An event sent near the end may not be
reported in any subsequent chunk if generation has already advanced.

Camera coordinates follow the server's Unity convention: +X right, +Y up,
+Z forward. Rotations are nonzero quaternions in `(x, y, z, w)` order, normalized
by the server. `target` gives a pose relative to session start and uses
`transition_chunks`; `velocity` gives a per-latent-step SE3 delta held until
replaced, and the server ignores `transition_chunks` in that mode. WASD strings
are supported in fixed scripts, not in live structural camera updates.

Explicit `--image`, `--prompt`, `--negative-prompt`, `--width`, `--height`,
`--fps`, `--seed`, and `--flow-shift` flags override the matching workload-file
values. `--num-chunks` and `--camera-pattern` apply only to the built-in
workload; when using `--workload`, set those fields in the file instead.
Camera controls cannot be injected through `extra_params`, which would bypass
the workload's validated mode and length.

Camera actions change what the world does, not what it costs, so `--camera-pattern`
(`forward`, `orbit`, `hold`) keeps a rollout representative rather than sweeping a
cost dimension.

### Interaction observations

Each session's JSON contains an `interactions` list. All observations use the
client's monotonic clock, with send timestamps relative to `session.start`:

- `send_to_queued_ms`: send start to the matching `session.interaction.queued` ACK.
- `send_to_first_reported_media_ms`: send start to the first nonempty media
  payload whose preceding metadata reports that event as started, active, or
  completed. Metadata arrival alone is not media arrival.
- `first_reported_media_chunk` and `first_reported_states`: the matching chunk
  and its reported states. Per-chunk event ID lists are also retained.
- Missing ACK or media observations remain `null`; `media_observation` is
  `not_observed` when no matching media arrived before the session ended.

ACK and media may arrive in either order. A reported event does not prove that
its action changed the pixels or that a browser displayed them. In particular,
the server can mark a replaced command as completed, even if a later command
superseded it before pose integration. Do not interpret absent completion as a
dropped event or these timings as motion-to-photon latency.

## Options worth knowing

| Flag | Purpose |
| --- | --- |
| `--target-fps` | Playback rate the real-time verdict is measured against, independent of the mux label. |
| `--sessions N` | N sequential measured rollouts; the report adds spread across them. |
| `--warmup-sessions N` | Unmeasured rollouts first, to pay compile and capture cost. |
| `--print-chunks` | One line per chunk as it arrives. |
| `--save-video PATH` | Remux the last measured session for eyeballing (needs vLLM-Omni). |
| `--ping-interval` | `session.ping` cadence keeping the server's stall clock fresh during a slow first chunk. `0` disables. |
| `--first-chunk-timeout` | Raise it when the server is still compiling or capturing graphs. |

## Output

`--output-json` writes the run's configuration, every chunk's arrival time and
size, per-session metrics, and the aggregate, along with the git revision and
hostname so a number can be traced back to what produced it.

## Tests

`tests/benchmarks/test_lingbot_world_realtime.py` covers the chunk and frame
arithmetic against the server's own formula, the metric math, and the protocol
handling against a scripted WebSocket server. It needs no GPU:

```bash
pytest tests/benchmarks/test_lingbot_world_realtime.py
```
