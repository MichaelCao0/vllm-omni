# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request state, recompute feedback, and sampler/logprob contracts."""

import gc
import weakref
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest
import torch
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata

from tests.model_executor.models.test_funaudiochat_native import _make_model_stub
from vllm_omni.model_executor.models.funaudiochat import funaudiochat as fac
from vllm_omni.model_executor.models.funaudiochat.sampler import FunAudioChatSampler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Sidecar:
    def __init__(self):
        self.calls = 0
        self.next_codes: torch.Tensor | None = None
        self.crq_past_key_values: torch.Tensor | None = None
        self.crq_generator: torch.Generator | None = None

    def crq_generate_forward(self, inputs_embeds, **kwargs):
        self.calls += 1
        previous = self.crq_past_key_values
        length = 0 if previous is None else int(previous.item())
        self.crq_past_key_values = torch.tensor(length + inputs_embeds.shape[1] * 2)
        self.crq_audio_embeds = torch.full((1, 4), float(self.calls))
        self.crq_generate_tokens = (
            torch.randint(1, 20, (1, 2), generator=self.crq_generator)
            if self.next_codes is None
            else self.next_codes.clone()
        )


def _make_replay_model(monkeypatch):
    model = _make_model_stub(group_size=2)
    model.audio_invert_tower = _Sidecar()
    model.get_language_model = lambda: SimpleNamespace(
        embed_input_ids=lambda ids: ids.float().reshape(-1, 1).expand(-1, 4).clone()
    )
    model.audio_tower = lambda codes: codes.float().sum(-1, keepdim=True).unsqueeze(-1).expand(-1, 1, 4)

    def logits(_model, hidden):
        scores = torch.zeros(hidden.shape[0], 100)
        scores[:, 7] = 1
        return scores

    monkeypatch.setattr(fac._NativeFunAudioChatBase, "compute_logits", logits)

    def sample(*, logits, sampling_metadata, forced_token_ids):
        tokens = logits.argmax(-1)
        for row, forced in enumerate(forced_token_ids):
            if forced is not None:
                tokens[row] = forced
        return SamplerOutput(sampled_token_ids=tokens.reshape(-1, 1), logprobs_tensors=None)

    model._ar_sampler = sample
    return model


@pytest.fixture
def replay_model(monkeypatch):
    return _make_replay_model(monkeypatch)


def _step(model, ids, offset, *, request_id="req", prompt_len=3):
    ids = torch.tensor(ids)
    _, embeds, update = model.preprocess(
        ids,
        None,
        request_id=request_id,
        _omni_num_computed_tokens=offset,
        _omni_prompt_len=prompt_len,
        _omni_seed=42,
    )
    logits = model.compute_logits(embeds[-1:])
    sampled = model.sample(logits, create_autospec(SamplingMetadata, instance=True))
    delta = model.postprocess(embeds, **update)["audio_token_ids"]
    return embeds, sampled.sampled_token_ids, delta


def test_pure_replay_preserves_inputs_crq_rng_and_emitted_audio(replay_model):
    model = replay_model
    prompt, _, _ = _step(model, [1, 2, 3], 0)
    first, _, first_delta = _step(model, [42], 3)
    second, _, _ = _step(model, [7], 4)
    expected = torch.cat([prompt, first, second])
    cache = model._crq_gpu_state["req"]["pkv"]
    rng = model._crq_generators["req"].get_state().clone()
    speech = model._speech_ids_gpu_state["req"]
    before = model.audio_invert_tower.calls
    state = dict(model._speech_state["req"])

    replayed, _, delta = _step(model, [1, 2, 3, 42, 7], 0)

    assert torch.equal(replayed, expected)
    assert torch.equal(second, (torch.full((1, 4), 7.0) + first_delta.sum()) / 2)
    assert model.audio_invert_tower.calls == before
    assert model._crq_gpu_state["req"]["pkv"] is cache
    assert model._speech_ids_gpu_state["req"] is speech
    assert torch.equal(model._crq_generators["req"].get_state(), rng)
    assert model._speech_state["req"] == state
    assert torch.all(delta == -1)
    assert model._replay_state["req"].processed_until == 5


def test_replay_plus_one_live_position_advances_sidecar_once(replay_model):
    model = replay_model
    prompt, _, _ = _step(model, [1, 2, 3], 0)
    first, _, _ = _step(model, [42], 3)
    before_calls = model.audio_invert_tower.calls
    before_len = model._speech_ids_gpu_state["req"].shape[1]

    embeds, _, delta = _step(model, [1, 2, 3, 42, 7], 0)

    assert torch.equal(embeds[:4], torch.cat([prompt, first]))
    assert model.audio_invert_tower.calls == before_calls + 1
    assert model._speech_ids_gpu_state["req"].shape[1] == before_len + 2
    assert delta.shape == (1, 2)
    assert torch.all(delta >= 0)
    assert model._replay_state["req"].processed_until == 5


class _InputDependentSidecar:
    """Small recurrent decoder whose output exposes incorrect request inputs.

    Unlike the RNG-only sidecar above, every codec group depends on the input,
    restored cache, previous audio embedding, speech history, and request RNG.
    This is a test decoder, not an approximation of the real CRQ transformer.
    """

    def __init__(self):
        self.crq_past_key_values: torch.Tensor | None = None
        self.crq_audio_embeds: torch.Tensor | None = None
        self.crq_generator: torch.Generator | None = None
        self.crq_speech_ids = torch.empty((1, 0), dtype=torch.long)

    def crq_generate_forward(self, inputs_embeds, **kwargs):
        inputs = inputs_embeds.sum(dim=1)
        past = self.crq_past_key_values
        previous_audio = self.crq_audio_embeds
        if past is None:
            past = torch.zeros_like(inputs)
        if previous_audio is None:
            previous_audio = torch.zeros_like(inputs)
        context = inputs + 0.25 * past + 0.125 * previous_audio + 0.001 * self.crq_speech_ids.float().sum()
        self.crq_past_key_values = torch.tanh(context / 128)
        self.crq_audio_embeds = inputs_embeds[:, -1].clone()
        draw = torch.randint(0, 17, (1, 2), generator=self.crq_generator)
        signal = (self.crq_past_key_values[:, :2] * 1000).round().long()
        # Stay below the test EOS (99), so termination cannot hide divergence.
        self.crq_generate_tokens = (signal + draw) % 89 + 1


def _make_input_dependent_model(monkeypatch):
    model = _make_replay_model(monkeypatch)
    model.audio_invert_tower = _InputDependentSidecar()
    return model


def _assert_same_request_state(reference, actual, request_id):
    assert torch.equal(reference._crq_gpu_state[request_id]["pkv"], actual._crq_gpu_state[request_id]["pkv"])
    assert torch.equal(reference._crq_gpu_state[request_id]["embeds"], actual._crq_gpu_state[request_id]["embeds"])
    assert torch.equal(reference._speech_ids_gpu_state[request_id], actual._speech_ids_gpu_state[request_id])
    assert torch.equal(
        reference._crq_generators[request_id].get_state(), actual._crq_generators[request_id].get_state()
    )
    assert reference._speech_state[request_id] == actual._speech_state[request_id]


@pytest.mark.parametrize("partial_replay_first", [False, True], ids=["replay-and-live", "pure-replay-then-live"])
def test_replay_live_and_following_decode_match_uninterrupted_conditioned_sidecar(monkeypatch, partial_replay_first):
    reference = _make_input_dependent_model(monkeypatch)
    resumed = _make_input_dependent_model(monkeypatch)
    original_embeds = []
    for ids, offset in (([1, 2, 3], 0), ([42], 3), ([7], 4)):
        expected = _step(reference, ids, offset)
        actual = _step(resumed, ids, offset)
        original_embeds.append(expected[0])
        for left, right in zip(expected, actual):
            assert torch.equal(left, right)

    if partial_replay_first:
        for ids, offset in (([1, 2, 3], 0), ([42, 7], 3)):
            _, _, delta = _step(resumed, ids, offset)
            assert torch.all(delta == -1)
            _assert_same_request_state(reference, resumed, "req")

    expected = _step(reference, [7], 5)
    replayed = _step(resumed, [1, 2, 3, 42, 7, 7], 0)
    assert torch.equal(replayed[0], torch.cat([*original_embeds, expected[0]]))
    assert torch.equal(replayed[1], expected[1])
    assert torch.equal(replayed[2], expected[2])
    _assert_same_request_state(reference, resumed, "req")

    for offset in (6, 7):
        expected = _step(reference, [7], offset)
        actual = _step(resumed, [7], offset)
        for left, right in zip(expected, actual):
            assert torch.equal(left, right)
        _assert_same_request_state(reference, resumed, "req")


def _batch_step(model, requests):
    entries = []
    for rid, ids, offset in requests:
        entries.append(
            model.preprocess(
                torch.tensor(ids),
                None,
                request_id=rid,
                _omni_prompt_len=3,
                _omni_num_computed_tokens=offset,
                _omni_seed=42,
            )
        )
    logits = model.compute_logits(torch.cat([entry[1][-1:] for entry in entries]))
    output = model.sample(logits, create_autospec(SamplingMetadata, instance=True))
    result = {}
    # Request-ID routing must tolerate filtered/reordered postprocess callbacks.
    for index in reversed(range(len(entries))):
        _, embeds, update = entries[index]
        delta = model.postprocess(embeds, **update)["audio_token_ids"]
        result[requests[index][0]] = (embeds, output.sampled_token_ids[index : index + 1], delta)
    return result


def test_reordered_live_replay_and_new_prefill_match_independent_request_trajectories(monkeypatch):
    reference = {rid: _make_input_dependent_model(monkeypatch) for rid in ("first", "second", "new")}
    batched = _make_input_dependent_model(monkeypatch)
    first_prompt = [1, 2, 3]
    second_prompt = [11, 12, 13]
    second_history = []
    for batch in (
        [("first", first_prompt, 0), ("second", second_prompt, 0)],
        [("first", [42], 3), ("second", [42], 3)],
    ):
        actual = _batch_step(batched, batch)
        for rid, ids, offset in batch:
            expected = _step(reference[rid], ids, offset, request_id=rid)
            for left, right in zip(expected, actual[rid]):
                assert torch.equal(left, right)
            _assert_same_request_state(reference[rid], batched, rid)
            if rid == "second":
                second_history.append(expected[0])

    # Two existing requests swap row order; one replays before its live input,
    # while a third request requires deferred prefill warmup in the same batch.
    actual = _batch_step(
        batched,
        [("second", [*second_prompt, 42, 7], 0), ("new", [21, 22, 23], 0), ("first", [7], 4)],
    )
    for rid, ids, offset in (("first", [7], 4), ("second", [7], 4), ("new", [21, 22, 23], 0)):
        expected = _step(reference[rid], ids, offset, request_id=rid)
        expected_embeds = torch.cat([*second_history, expected[0]]) if rid == "second" else expected[0]
        assert torch.equal(actual[rid][0], expected_embeds)
        assert torch.equal(actual[rid][1], expected[1])
        assert torch.equal(actual[rid][2], expected[2])
        _assert_same_request_state(reference[rid], batched, rid)

    batch = [("first", [7], 5), ("new", [42], 3), ("second", [7], 5)]
    actual = _batch_step(batched, batch)
    for rid, ids, offset in batch:
        expected = _step(reference[rid], ids, offset, request_id=rid)
        for left, right in zip(expected, actual[rid]):
            assert torch.equal(left, right)
        _assert_same_request_state(reference[rid], batched, rid)


@pytest.mark.parametrize("replay_ids", [[1, 2, 3, 42], [1, 2, 3, 42, 7]], ids=["pure", "with-live-position"])
def test_first_replay_logs_position_range_once(replay_model, monkeypatch, replay_ids):
    messages = []
    monkeypatch.setattr(fac.logger, "info", lambda message, *args: messages.append(message % args))
    _step(replay_model, [1, 2, 3], 0)
    _step(replay_model, [42], 3)
    assert not messages

    _step(replay_model, replay_ids, 0)
    _step(replay_model, replay_ids, 0)

    assert messages == [
        f"FunAudioChat stage 0 replay: request_id=req, scheduled=[0, {len(replay_ids)}), "
        "replayed=[0, 4), processed_until=4"
    ]


def test_replay_log_is_per_request_and_resets_after_cleanup(replay_model, monkeypatch):
    messages = []
    monkeypatch.setattr(fac.logger, "info", lambda message, *args: messages.append(message % args))
    for rid in ("first", "second"):
        _step(replay_model, [1, 2, 3], 0, request_id=rid)
        _step(replay_model, [1, 2], 0, request_id=rid)
    assert len(messages) == 2
    assert "request_id=first" in messages[0]
    assert "request_id=second" in messages[1]

    replay_model.on_requests_finished({"first"})
    _step(replay_model, [1, 2, 3], 0, request_id="first")
    assert len(messages) == 2
    _step(replay_model, [2, 3], 1, request_id="first")
    assert len(messages) == 3
    assert messages[-1] == (
        "FunAudioChat stage 0 replay: request_id=first, scheduled=[1, 3), replayed=[1, 3), processed_until=3"
    )


def test_replay_after_audio_eos_preserves_historical_speech_feedback(replay_model):
    model = replay_model
    _step(model, [1, 2, 3], 0)
    _step(model, [42], 3)
    model.audio_invert_tower.next_codes = torch.tensor([[99, 99]])
    speech_embeds, eos, _ = _step(model, [7], 4)
    assert eos.item() == 99
    assert model._speech_state["req"][fac._GENERATE_SPEECH_KEY] is False
    text_embeds, _, _ = _step(model, [99], 5)
    before = model.audio_invert_tower.calls

    replayed, _, delta = _step(model, [7, 99], 4)

    assert torch.equal(replayed, torch.cat([speech_embeds, text_embeds]))
    assert torch.equal(text_embeds, torch.full((1, 4), 99.0))
    assert model.audio_invert_tower.calls == before
    assert torch.all(delta == -1)


def test_replay_before_audio_bos_does_not_reinitialize_prompt_sidecar(replay_model):
    model = replay_model
    model.sp_gen_kwargs["force_text_abos"] = False
    prompt, _, _ = _step(model, [1, 2, 3], 0)
    text, _, _ = _step(model, [7], 3)
    rng = model._crq_generators["req"].get_state().clone()
    calls = model.audio_invert_tower.calls

    replayed, _, delta = _step(model, [2, 3, 7], 1)

    assert torch.equal(replayed, torch.cat([prompt[1:], text]))
    assert model.audio_invert_tower.calls == calls
    assert torch.equal(model._crq_generators["req"].get_state(), rng)
    assert torch.all(delta == -1)


def test_text_only_progress_and_replay_do_not_require_downstream_postprocess(replay_model):
    model = replay_model

    def text_step(ids, offset, final_stage=None):
        kwargs = {
            "request_id": "text",
            "_omni_prompt_len": 3,
            "_omni_num_computed_tokens": offset,
            "_omni_seed": 42,
        }
        if final_stage is not None:
            kwargs["omni_final_stage_id"] = final_stage
        _, embeds, _ = model.preprocess(torch.tensor(ids), None, **kwargs)
        logits = model.compute_logits(embeds[-1:])
        result = model.sample(logits, create_autospec(SamplingMetadata, instance=True))
        # Match the runner: no postprocess call when the final stage is 0.
        return embeds, result.sampled_token_ids

    prompt, tokens = text_step([1, 2, 3], 0, final_stage=0)
    assert tokens.tolist() == [[7]]  # No forced audio BOS.
    assert model._replay_state["text"].processed_until == 3
    decode, _ = text_step([7], 3)
    assert model._replay_state["text"].processed_until == 4
    replay, _ = text_step([1, 2, 3, 7], 0)
    assert torch.equal(replay, torch.cat([prompt, decode]))
    assert model._replay_state["text"].processed_until == 4
    assert model.audio_invert_tower.calls == 0
    assert "text" not in model._crq_gpu_state
    assert "text" not in model._speech_ids_gpu_state
    assert "text" not in model._crq_generators
    assert "text" not in model._speech_state


def test_mixed_text_only_and_audio_prefill_routes_warmup_by_request_id(replay_model):
    model = replay_model
    entries = []
    for rid, final_stage in (("text", 0), ("audio", 1)):
        entries.append(
            model.preprocess(
                torch.tensor([1, 2, 3]),
                None,
                request_id=rid,
                _omni_prompt_len=3,
                _omni_num_computed_tokens=0,
                _omni_seed=42,
                omni_final_stage_id=final_stage,
            )
        )
    logits = model.compute_logits(torch.cat([entry[1][-1:] for entry in entries]))
    output = model.sample(logits, create_autospec(SamplingMetadata, instance=True))
    assert output.sampled_token_ids.tolist() == [[7], [42]]
    assert model._replay_state["text"].processed_until == 3
    assert model._replay_state["audio"].processed_until == 0

    # The runner filters the text-only row before model.postprocess.
    model.postprocess(entries[1][1], **entries[1][2])

    assert model.audio_invert_tower.calls == 1
    assert model._replay_state["audio"].processed_until == 3
    assert "text" not in model._crq_gpu_state
    assert "audio" in model._crq_gpu_state


def test_replay_isolated_from_other_request_and_reordered_postprocess(replay_model):
    model = replay_model
    _step(model, [1, 2, 3], 0, request_id="replay")
    _step(model, [1, 2, 3], 0, request_id="live")
    before_rng = model._crq_generators["replay"].get_state().clone()
    before_cache = model._crq_gpu_state["replay"]["pkv"]
    replay_inputs = model.preprocess(
        torch.tensor([1, 2]), None, request_id="replay", _omni_prompt_len=3, _omni_num_computed_tokens=0
    )
    live_inputs = model.preprocess(
        torch.tensor([42]), None, request_id="live", _omni_prompt_len=3, _omni_num_computed_tokens=3
    )
    logits = model.compute_logits(torch.cat([replay_inputs[1][-1:], live_inputs[1][-1:]]))
    model.sample(logits, create_autospec(SamplingMetadata, instance=True))
    live_delta = model.postprocess(live_inputs[1], **live_inputs[2])["audio_token_ids"]
    replay_delta = model.postprocess(replay_inputs[1], **replay_inputs[2])["audio_token_ids"]

    assert torch.all(live_delta >= 0)
    assert torch.all(replay_delta == -1)
    assert model._crq_gpu_state["replay"]["pkv"] is before_cache
    assert torch.equal(model._crq_generators["replay"].get_state(), before_rng)
    assert model._replay_state["live"].processed_until == 4
    assert model._replay_state["replay"].processed_until == 3


@pytest.mark.parametrize("offset,ids", [(5, [7]), (3, [7, 8])])
def test_replay_rejects_missing_or_multiple_live_positions(replay_model, offset, ids):
    _step(replay_model, [1, 2, 3], 0)
    with pytest.raises(RuntimeError, match="beyond"):
        _step(replay_model, ids, offset)


def test_replay_rejects_missing_feedback_instead_of_using_latest_codes(replay_model):
    _step(replay_model, [1, 2, 3], 0)
    _step(replay_model, [42], 3)
    replay_model._replay_state["req"].feedback.clear()
    with pytest.raises(RuntimeError, match="lacks codec feedback"):
        _step(replay_model, [42], 3)


def test_finished_request_releases_tower_batch_and_prompt_references(replay_model):
    model = replay_model
    _step(model, [1, 2, 3], 0, request_id="other")
    _step(model, [1, 2, 3], 0)
    _step(model, [42], 3)
    cache_ref = weakref.ref(model._crq_gpu_state["req"]["pkv"])
    history_ref = weakref.ref(model._speech_ids_gpu_state["req"])
    prompt_ref = weakref.ref(model._replay_state["req"].prompt_embeds)

    model.on_requests_finished({"req"})
    gc.collect()

    assert cache_ref() is None
    assert history_ref() is None
    assert prompt_ref() is None
    assert model.audio_invert_tower.crq_past_key_values is None
    assert model.audio_invert_tower.crq_speech_ids is None
    assert not model._batch_req_infos
    assert not model._batch_sidecar_results
    assert "other" in model._crq_gpu_state
    assert "other" in model._replay_state


def _sampling_metadata(logprobs=2):
    class StopMask:
        def apply(self, logits):
            # Represents a min_tokens/bad-word-style mask running after
            # compute_logits. The model boundary must survive this mask.
            logits[:, [42, 99]] = -torch.inf
            return logits

    allowed = torch.zeros(3, 100, dtype=torch.bool)
    allowed[:, [42, 99]] = True
    return SamplingMetadata(
        temperature=None,
        all_greedy=True,
        all_random=False,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=logprobs,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(3),
        presence_penalties=torch.zeros(3),
        repetition_penalties=torch.ones(3),
        output_token_ids=[[], [], []],
        allowed_token_ids_mask=allowed,
        bad_words_token_ids={},
        logitsprocs=create_autospec(
            LogitsProcessors, instance=True, non_argmax_invariant=[StopMask()], argmax_invariant=[]
        ),
    )


@pytest.mark.parametrize("mode", ["raw_logprobs", "processed_logprobs"])
def test_forced_boundaries_keep_selected_token_logprobs_consistent(mode):
    sampler = FunAudioChatSampler(logprobs_mode=mode)
    logits = torch.zeros(3, 100)
    logits[:, 7] = 3
    original = logits.clone()
    result = sampler(logits, _sampling_metadata(), forced_token_ids=[42, 99, None])

    assert result.sampled_token_ids.tolist() == [[42], [99], [7]]
    logprobs = result.logprobs_tensors
    assert logprobs.logprob_token_ids[:, 0].tolist() == [42, 99, 7]
    assert torch.isfinite(logprobs.logprobs[:, 0]).all()
    if mode == "raw_logprobs":
        expected = original.log_softmax(-1)[torch.arange(3), torch.tensor([42, 99, 7])]
        assert torch.allclose(logprobs.logprobs[:, 0], expected)
    else:
        assert logprobs.logprobs[:2, 0].tolist() == [0.0, 0.0]
    assert sampler._forced_token_ids is None


def test_sampler_rejects_misaligned_force_mask():
    with pytest.raises(ValueError, match="sampling batch"):
        FunAudioChatSampler()(torch.zeros(3, 100), _sampling_metadata(), forced_token_ids=[42])
