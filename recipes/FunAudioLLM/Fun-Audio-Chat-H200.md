# Fun-Audio-Chat speech conversation — H200

## Summary

- Vendor: FunAudioLLM
- Model: `FunAudioLLM/Fun-Audio-Chat-8B`
- Audio decoder: `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`
- Task: Complete-turn speech input to text and generated speech
- Mode: OpenAI-compatible Chat Completions with optional streaming output
- Hardware: Two NVIDIA H200 GPUs, one per pipeline stage
- Maintainer: Community
- Qualification: Bounded remote functional validation; see the evidence and limits below

## Supported model contract

This profile consumes a complete audio recording before generating a response.
It streams generated codec tokens between the two stages so audio can reach
the client while the response is being generated.

| Contract | This integration |
| --- | --- |
| Input | One complete audio recording per request; a six-second WAV plus synthetic 30/300-second capacity probes were exercised |
| Request | `/v1/chat/completions`, `input_audio`, `modalities: ["text", "audio"]` |
| System prompt | Official `SPOKEN_S2M_PROMPT`, which requests joint text and speech and a concise conversational reply |
| Output | Text plus mono, 24 kHz generated audio; duration depends on the response |
| Output streaming | `stream: true` uses Server-Sent Events; `stream: false` returns the completed response |
| Voice | The upstream default speaker embedding in `utils/new_spk2info.pt` |
| Continuous audio input / duplex | Not implemented by this profile |

| Provided profile | Placement and precision | Qualification status |
| --- | --- | --- |
| Two-stage CUDA | GPU 0: native vLLM AR and CRQ sidecar, BF16; GPU 1: CosyVoice3 decoder, FP32 flow | Online and offline generation exercised; four overlapping requests and cancellation/recovery exercised |

Both stages use eager execution and `max_num_seqs: 4`. This is pipeline
placement across two devices, with tensor parallelism 1 in each stage.

## References

- [Official Fun-Audio-Chat source](https://github.com/FunAudioLLM/Fun-Audio-Chat)
- [Model card](https://huggingface.co/FunAudioLLM/Fun-Audio-Chat-8B)
- [CosyVoice3 checkpoint](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
- [Chat Completions API](../../docs/serving/chat_completions_api.md)
- [Supported models](../../docs/models/supported_models.md) and
  [feature compatibility](../../docs/user_guide/feature_compatibility.md)
- [Deploy configuration](../../vllm_omni/deploy/funaudiochat.yaml) and
  [pipeline topology](../../vllm_omni/model_executor/models/funaudiochat/pipeline.py)
- [Online integration tests](../../tests/e2e/online_serving/test_fun_audio_chat.py)
- Related: [issue #7244](https://github.com/vllm-project/vllm-omni/issues/7244)

The integration builds on TheBasy's
[PR #5862](https://github.com/vllm-project/vllm-omni/pull/5862), at commit
`fc06f6866082f83f9196b971dbf346eed22dfb96`, with migration to the current
pipeline and runner contracts. The continuous-input and bidirectional-streaming
requests in #7244 remain outside this profile.

## Hardware and software

The target profile uses two devices from a four-H200 remote workspace. Each
device exposes 143,771 MiB, and the device topology reports NV18 links between
each pair. The remaining two devices were used for independent validation. Other
device counts and accelerator families are not qualified by this recipe.

| Software | Integration environment |
| --- | --- |
| OS | Ubuntu 24.04.3 |
| Python | 3.11.16 |
| NVIDIA driver | 570.124.06 |
| PyTorch | 2.13.0+cu129 |
| CUDA compiler | 12.9.86 for the FlashAttention-2 build |
| vLLM | 0.29.0+cu129 |
| Transformers | 5.14.1 |
| FlashAttention-2 | 2.8.3.post1, built against the serving environment |
| vLLM-Omni base | `154a1f25f537c19180ce4abfdcf3163db7c57956` plus this integration |

Use the installation instructions for your vLLM-Omni checkout. The integration
requires a vLLM build containing `vllm.model_executor.models.funaudiochat`.
The official model's reference environment is separate from the vLLM-Omni
environment; installing its full requirements over the serving environment can
replace compatible PyTorch and Transformers versions.

The independent qualification environment used the CUDA 12.9 vLLM wheel and
the repository's CUDA dependencies. The following reproduces that dependency
selection in a fresh environment (run from this checkout):

```bash
uv venv --python 3.11 --seed .venv-funaudiochat
source .venv-funaudiochat/bin/activate
cat > /tmp/funaudiochat-constraints.txt <<'EOF'
torch==2.13.0+cu129
torchvision==0.28.0+cu129
torchaudio==2.11.0+cu129
transformers==5.14.1
EOF
uv pip install --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  -c /tmp/funaudiochat-constraints.txt \
  'vllm @ https://github.com/vllm-project/vllm/releases/download/v0.29.0/vllm-0.29.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl' \
  -r requirements/cuda.txt
uv pip install --no-deps -e .
uv pip check
export OMP_NUM_THREADS=4
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

The audio dependency import path also imports OpenCV. On a minimal Ubuntu
image, its native libraries include `libgl1` and `libxcb1`; ensure those and
their dependencies are present. Qualification used verified Ubuntu 24.04
packages in a task-private library prefix. Advanced tests additionally need
`ffmpeg`, the repository's pytest dependencies, and `opencc` for text comparison.

## Checkpoint and source setup

Run these commands from the vLLM-Omni repository root. Keep the pinned official
source available to both worker processes: the CRQ decoder and default speaker
asset are loaded from it. The production audio decoder otherwise reuses the
in-tree CosyVoice3 implementation.

```bash
mkdir -p .models
git clone https://github.com/FunAudioLLM/Fun-Audio-Chat.git .models/Fun-Audio-Chat
git -C .models/Fun-Audio-Chat checkout 8ba984b64b4880918db2807e08d795475481200c
export FUN_AUDIO_CHAT_HOME="$PWD/.models/Fun-Audio-Chat"
export PYTHONPATH="$PWD:$FUN_AUDIO_CHAT_HOME:${PYTHONPATH:-}"
export FUNAUDIOCHAT_MODEL_DIR="$PWD/.models/Fun-Audio-Chat-8B"
export FUNAUDIOCHAT_CODEC_DIR="$PWD/.models/Fun-CosyVoice3-0.5B-2512"

hf download FunAudioLLM/Fun-Audio-Chat-8B \
  --revision 7bf72dc7c705493f817178a0859efff91e9cf73c \
  --local-dir "$FUNAUDIOCHAT_MODEL_DIR"
hf download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --revision 29e01c4e8d000f4bcd70751be16fa94bf3d85a18 \
  --local-dir "$FUNAUDIOCHAT_CODEC_DIR"
```

`FUN_AUDIO_CHAT_SPK_INFO` can point to the same upstream `new_spk2info.pt` asset
when it is stored separately. This path override does not provide a voice
cloning API.

### FlashAttention-2 for native audio profiling

The native vLLM audio tower requires the external `flash_attn` package for its
default long-audio memory profile. Install it into the same environment as
vLLM-Omni, using a CUDA toolkit compatible with the installed PyTorch build.
The vLLM 0.29.0 serving image includes `nvcc`; a separate virtual environment
also needs access to a complete CUDA toolkit. The H200 qualification environment
uses CUDA 12.9.86 with PyTorch 2.13.0+cu129.

The following builds pinned FlashAttention 2.8.3.post1 source against the
environment's current PyTorch and C++ ABI. It restricts compilation to Hopper
for this H200 profile. `--no-deps` preserves the serving environment's PyTorch,
and `--no-cache-dir` avoids reusing a local wheel built against another Torch
version. Leave the ABI override unset.

```bash
python -c "from pathlib import Path; from torch.utils.cpp_extension import CUDA_HOME; assert CUDA_HOME and (Path(CUDA_HOME) / 'bin/nvcc').is_file(), 'A CUDA toolkit with nvcc is required'"
python -m pip install --no-deps packaging psutil ninja setuptools wheel
export FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=90
export MAX_JOBS=4 NVCC_THREADS=2
python -m pip install --no-build-isolation --no-deps --no-cache-dir --force-reinstall \
  'https://files.pythonhosted.org/packages/01/7a/92a46e7cd6bbb4d7b2855a457c3b855df54a97af5656d98fc92e58e61065/flash_attn-2.8.3.post1.tar.gz#sha256=55d5103ed846da8b56e0797acf4bde07dee4b1c7e8907fcfc6699c203030c348'
python -c "import flash_attn; from funaudiochat.modeling_funaudiochat import FunAudioChatDecoder; print(flash_attn.__version__)"
```

Allow time for CUDA compilation before model startup. The CI jobs reserve
90 minutes for dependency preparation and model validation together; this is
not an inference latency measurement. The remote build and a real CUDA BF16
FlashAttention forward check passed. A worker snapshot confirmed the native
audio encoder uses `MMEncoderAttention` with `FLASH_ATTN`; the external package
requirement and the encoder's selected backend were checked separately.

## Serve and send a request

The following uses the default two-device launch configuration. Qualification
used the same stage settings, with only physical GPU placement and local
checkpoint paths changed.

```bash
export FUNAUDIOCHAT_DEPLOY_CONFIG="$PWD/.models/funaudiochat-local.yaml"
python - <<'PYCONFIG'
import os
from pathlib import Path
import yaml

config = yaml.safe_load(Path("vllm_omni/deploy/funaudiochat.yaml").read_text())
config["stages"][1]["model"] = os.environ["FUNAUDIOCHAT_CODEC_DIR"]
Path(os.environ["FUNAUDIOCHAT_DEPLOY_CONFIG"]).write_text(yaml.safe_dump(config, sort_keys=False))
PYCONFIG
vllm serve "$FUNAUDIOCHAT_MODEL_DIR" --omni \
  --deploy-config "$FUNAUDIOCHAT_DEPLOY_CONFIG" \
  --host 127.0.0.1 --port 8092 --served-model-name funaudiochat
```

The native vLLM audio tower profiles a long synthetic audio input by default.
At the recorded vLLM revision, this path requires FlashAttention-2. The final
deployment and repository tests use the native 300-second memory profile,
without a profiling-length override. A profiling length controls synthetic
startup memory estimation; it does **not** enforce an input-duration limit.
The 30/300-second probes repeat the six-second sample to test capacity; they
do not establish quality on long conversations or a maximum supported length.

Once `/health` returns HTTP 200, use the official sample with the standard Chat
API. The request loads the official concise conversational persona from the
pinned source, matching the normal online and offline tests. Set
`"stream": False` in the request builder for a completed response.

```bash
python - <<'PY' > funaudiochat-request.json
import base64
import json
import os
import runpy
from pathlib import Path

source = Path(os.environ["FUN_AUDIO_CHAT_HOME"])
sample = source / "examples" / "ck7vv9ag.wav"
system_prompt = runpy.run_path(str(source / "utils" / "constant.py"))["SPOKEN_S2M_PROMPT"]
request = {
    "model": "funaudiochat",
    "messages": [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": [{
                "type": "input_audio",
                "input_audio": {
                    "data": base64.b64encode(sample.read_bytes()).decode(),
                    "format": "wav",
                },
            }],
        },
    ],
    "modalities": ["text", "audio"],
    "stream": True,
    "temperature": 0.0,
    "repetition_penalty": 1.0,
    "max_tokens": 512,
    "seed": 42,
}
print(json.dumps(request))
PY
curl --no-buffer --fail-with-body http://127.0.0.1:8092/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @funaudiochat-request.json
```

The request's repetition penalty applies to text generation. CRQ speech
sampling uses the model's separate defaults: temperature 0.8, top-p 0.9,
top-k 0, and repetition penalty 1.2. Do not treat the text seed or greedy
decoding setting as a claim of bit-identical waveform output across runtimes.

## Offline complete output

After stopping the online server, reuse the same environment and deploy YAML
with the shared `Omni` Python entrypoint. This validated example consumes the
official audio sample and writes the complete output to a WAV file. Keep the
main guard when saving it as a script because the workers use multiprocessing.

```python
import os
from pathlib import Path
import runpy

import soundfile as sf
import torch

from vllm_omni.entrypoints.omni import Omni


def main():
    source = Path(os.environ["FUN_AUDIO_CHAT_HOME"])
    system = runpy.run_path(str(source / "utils/constant.py"))["SPOKEN_S2M_PROMPT"]
    audio, sr = sf.read(source / "examples/ck7vv9ag.wav", dtype="float32")
    prompt = (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        "<|im_start|>user\n<|audio_bos|><|AUDIO|><|audio_eos|><|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    omni = Omni(
        model=os.environ["FUNAUDIOCHAT_MODEL_DIR"],
        deploy_config=os.environ["FUNAUDIOCHAT_DEPLOY_CONFIG"],
        trust_remote_code=True,
    )
    try:
        params = [item.clone() for item in omni.default_sampling_params_list]
        params[0].max_tokens = 512
        outputs = omni.generate(
            [{"prompt": prompt, "multi_modal_data": {"audio": (audio, sr)},
              "modalities": ["text", "audio"]}],
            sampling_params_list=params,
        )
        for output in outputs:
            if output.final_output_type == "text":
                print(output.outputs[0].text)
            elif output.final_output_type == "audio":
                payload = output.outputs[0].multimodal_output
                raw = payload["audio"]
                chunks = raw if isinstance(raw, list) else [raw]
                waveform = torch.cat([torch.as_tensor(chunk).detach().float().cpu().reshape(-1) for chunk in chunks])
                sf.write("funaudiochat-output.wav", waveform.numpy(), 24000)
                print(f"Saved {waveform.numel() / 24000:.2f}s to funaudiochat-output.wav")
    finally:
        omni.close()


if __name__ == "__main__":
    main()
```

## Verification and evidence

From the repository root, the shared test runtime starts the server and checks
offline output, non-streaming/streaming HTTP, and four mixed concurrent requests:

```bash
export FUNAUDIOCHAT_MODEL="$FUNAUDIOCHAT_MODEL_DIR"
pytest -s -v tests/e2e/offline_inference/test_fun_audio_chat.py \
  tests/e2e/online_serving/test_fun_audio_chat.py \
  -m 'advanced_model and cuda' --run-level advanced_model
```

For a local decoder checkpoint, set `FUNAUDIOCHAT_DEPLOY_CONFIG` to a copy of
the deploy YAML whose stage 1 `model` points to `FUNAUDIOCHAT_CODEC_DIR`.
Expected output is nonempty text and finite, nonempty 24 kHz mono audio.
Streaming validation must also establish that chunks contain only new samples,
the final audio tail is delivered, and concurrent requests remain isolated.

| Check | Recorded evidence |
| --- | --- |
| Official reference | Six-second `examples/ck7vv9ag.wav` generated 415 codec tokens and 398,400 audio samples at 24 kHz (16.6 seconds) |
| Reference input SHA-256 | `680385c361ba4a08d8e09e26ddee34d025313ffe7aba61c9d8ee10c68c64175d` |
| Reference software | PyTorch 2.8.0+cu128; Transformers 4.52.3; seed 42; at most 512 generated tokens |
| Default six-second input | Completed and streaming responses both produced 16.0 seconds of speech; streaming delivered nine nonempty audio chunks |
| Whisper-small comparison | Generated text vs audio transcript similarity: 0.9797 for both default HTTP modes; official reference and fixed-codec decodes: 0.9868; repository threshold: greater than 0.8 |
| Fixed-codec completeness | The same 415 codec tokens decoded to 398,400 samples in both full and streaming modes, including the final tail |
| First nonempty audio / request completion | Six-second input: 1.088 / 10.608 seconds; synthetic 30-second input: 1.226 / 14.610; synthetic 300-second input: 1.389 / 14.594 |
| Peak device memory during the above HTTP probes | NVML used memory sampled every 0.2 seconds, including reserved memory: stage 0 up to 98,468,626,432 bytes; stage 1 up to 22,069,379,072 bytes |
| Completion and cancellation lifecycle | Three long/short/abort/recovery cycles; per-request model and runner state empty at idle; allocated memory growth zero on both stages, without `gc` or `empty_cache` |
| HTTP cancellation | Four requests overlapped; one disconnected after its first audio chunk, three continued to natural completion, and a new request then completed |

These are single-run functional observations in a shared CPU environment,
not controlled performance results. The timing scope is a sequential request
after server startup, with memory monitoring enabled. First-request warmup
and reserved allocations can affect the observed values. The six-second
input's non-streaming request took 12.511 seconds. Automated transcription
does not substitute for human listening; human listening and browser playback
were not part of these checks.

Under an artificial 33-block KV budget, four fixed English passages completed
with three observed scheduler preemptions and three model replay records.
All four transcript similarities exceeded 0.8. One pressure-run utterance
repeated its last phrase (score 0.8921); the same four prompts without pressure
scored 1.0 and did not repeat that phrase. The pressure result had 140 completion
tokens and 27.0 seconds of audio, versus 116 tokens and 22.2 seconds in the
four-request control, or 117 tokens and 22.4 seconds in a separate single-request
control. These are 24 and 23 additional AR steps, respectively, with the
corresponding additional audio samples.
This is a generation-content difference; the cause is not established by the
test. Do not interpret replay support as identical speech across batch shapes
or as a guarantee against generated repetition. CRQ KV remains resident on the
GPU until request cleanup; replay rebuilds the language model's paged KV only.

## Supported features

| Feature | Status in this profile |
| --- | --- |
| Complete-turn speech input | Exercised with the official sample and synthetic capacity probes |
| [Streaming output](../../docs/serving/chat_completions_api.md) | Exercised through inter-stage async chunks and Chat SSE |
| Four concurrent requests | Exercised with distinct prompts, cancellation, and recovery |
| Continuous input, barge-in, bidirectional audio | Not implemented |
| `/v1/audio/speech` and voice cloning | Not exposed by this integration |
| Prefix caching and chunked prefill | Disabled in the provided deploy configuration |
| CUDA graphs and asynchronous scheduling | Disabled in the provided deploy configuration |
| Tensor parallelism greater than 1, quantization, other accelerators | Not qualified |
