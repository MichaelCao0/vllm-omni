# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from typing import cast
from unittest.mock import create_autospec

import pytest
import torch
from transformers.generation.logits_process import (
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopPLogitsWarper,
)
from transformers.modeling_outputs import BaseModelOutput
from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata

import vllm_omni.model_executor.models.funaudiochat.funaudiochat as fac_mod
from vllm_omni.model_executor.models.funaudiochat.funaudiochat import (
    DEFAULT_SP_GEN_KWARGS,
    FunAudioChatForConditionalGeneration,
)
from vllm_omni.transformers_utils.configs.funaudiochat import FunAudioChatConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_model_stub(
    *,
    audio_bos_id: int = 42,
    audio_eos_id: int = 99,
    group_size: int = 5,
    hidden_size: int = 4,
):
    model = object.__new__(FunAudioChatForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.config = FunAudioChatConfig(
        audio_config={"group_size": group_size, "eos_token_id": audio_eos_id},
        text_config={
            "model_type": "qwen3",
            "hidden_size": hidden_size,
            "audio_bos_index": audio_bos_id,
            "audio_eos_index": audio_eos_id,
        },
    )
    model.sp_gen_kwargs = DEFAULT_SP_GEN_KWARGS.copy()
    model._batch_preprocess_in_progress = False
    model._batch_req_infos = []
    model._batch_sidecar_results = []
    model._postprocess_cursor = 0
    model._logged_stage0_backend = True
    model._speech_state = {}
    model._crq_gpu_state = {}
    model._speech_ids_gpu_state = {}
    model._crq_generators = {}
    model._replay_state = {}
    model._crq_bound_req_id = None
    model._logprobs_mode = "raw_logprobs"
    model.get_language_model = lambda: SimpleNamespace(
        embed_input_ids=lambda input_ids: torch.zeros(
            (input_ids.reshape(-1).numel(), hidden_size),
            dtype=torch.float32,
            device=input_ids.device,
        )
    )
    model.audio_tower = lambda audio_ids: BaseModelOutput(
        last_hidden_state=torch.full(
            (audio_ids.shape[0], 1, hidden_size),
            2.0,
            dtype=torch.float32,
            device=audio_ids.device,
        )
    )
    model._get_stage0_backend = lambda: "TEST"
    return model


def test_default_sp_gen_kwargs_match_official_defaults():
    assert DEFAULT_SP_GEN_KWARGS == {
        "text_greedy": True,
        "only_crq_sampling": True,
        "disable_speech": False,
        "force_text_abos": True,
    }


def test_load_weights_packs_native_audio_qkv_and_keeps_crq_weights(monkeypatch):
    model = _make_model_stub()
    packed = torch.nn.Parameter(torch.zeros(6, 2))
    shards = {"q": 0, "k": 1, "v": 2}

    def load_shard(param, value, shard):
        with torch.no_grad():
            param[2 * shards[shard] : 2 * (shards[shard] + 1)].copy_(value)

    packed.weight_loader = load_shard
    prefix = "continuous_audio_tower.layers.0.self_attn."
    model.named_parameters = lambda: iter([(prefix + "qkv_proj.weight", packed)])
    remaining = []

    class Loader:
        def __init__(self, module):
            assert module is model

        def load_weights(self, weights):
            remaining.extend(weights)
            return {name for name, _ in remaining}

    monkeypatch.setattr(fac_mod, "AutoWeightsLoader", Loader)
    weights = [(prefix + f"{shard}_proj.weight", torch.full((2, 2), float(i + 1))) for i, shard in enumerate(shards)]
    crq_name = "audio_invert_tower.crq_transformer.layers.0.self_attn.q_proj.weight"
    weights.append((crq_name, torch.ones(2, 2)))

    loaded = model.load_weights(iter(weights))

    assert packed[:, 0].tolist() == [1, 1, 2, 2, 3, 3]
    assert [name for name, _ in remaining] == [crq_name]
    assert loaded == {prefix + "qkv_proj.weight", crq_name}


def test_pooler_output_buffer_only_snapshots_incremental_audio_groups():
    assert FunAudioChatForConditionalGeneration.pooler_output_buffer_keys == ("audio_token_ids",)


def test_build_crq_sampling_config_matches_official_sampling_defaults():
    model = _make_model_stub()
    processors, do_sample = model._build_crq_sampling_config()
    assert do_sample is True
    assert [type(p) for p in processors] == [
        RepetitionPenaltyLogitsProcessor,
        TemperatureLogitsWarper,
        TopPLogitsWarper,
    ]
    assert processors[0].penalty == 1.2
    assert processors[1].temperature == 0.8
    assert processors[2].top_p == 0.9


def test_crq_defaults_do_not_depend_on_text_greedy_setting():
    model = _make_model_stub()
    model.sp_gen_kwargs["text_greedy"] = False
    processors, do_sample = model._build_crq_sampling_config()
    assert do_sample is True
    assert processors[0].penalty == 1.2
    assert processors[1].temperature == 0.8
    assert processors[2].top_p == 0.9


def test_postprocess_sampled_tokens_commits_audio_bos_without_mutating_tokens():
    model = _make_model_stub()
    tokens = torch.tensor([42])
    result = model.postprocess_sampled_tokens(tokens, ["req0"], {"req0": 0}, {"req0": {}})
    assert result is tokens
    assert tokens.tolist() == [42]
    assert model._speech_state["req0"][fac_mod._GENERATE_SPEECH_KEY] is True
    assert model._speech_state["req0"][fac_mod._FORCE_AUDIO_BOS_KEY] is False


def test_postprocess_sampled_tokens_preserves_regular_sampler_token():
    model = _make_model_stub()
    tokens = torch.tensor([7])
    assert model.postprocess_sampled_tokens(tokens, ["req0"], {"req0": 0}, {}) is tokens
    assert tokens.tolist() == [7]
    assert model._speech_state["req0"][fac_mod._GENERATE_SPEECH_KEY] is False


def test_postprocess_does_not_rewrite_sampler_output_even_if_force_flag_is_present():
    model = _make_model_stub()
    tokens = torch.tensor([7])
    step = {"req0": {fac_mod._FORCE_AUDIO_BOS_KEY: True}}
    model.postprocess_sampled_tokens(tokens, ["req0"], {"req0": 0}, step)
    assert tokens.tolist() == [7]
    assert model._speech_state["req0"][fac_mod._GENERATE_SPEECH_KEY] is False


def test_postprocess_keeps_speech_active_for_text_audio_eos_without_codec_eos():
    model = _make_model_stub()
    model._speech_state["req0"] = {fac_mod._GENERATE_SPEECH_KEY: True}
    tokens = torch.tensor([99])
    step = {"req0": {fac_mod._FINISH_SPEECH_KEY: False}}

    assert model.postprocess_sampled_tokens(tokens, ["req0"], {"req0": 0}, step) is tokens
    assert tokens.tolist() == [99]
    assert model._speech_state["req0"][fac_mod._GENERATE_SPEECH_KEY] is True


@pytest.mark.parametrize("sampled_token", [7, 42, 99], ids=["text", "audio-bos", "audio-eos"])
def test_postprocess_codec_eos_finishes_speech_without_rewriting_sampled_token(sampled_token):
    model = _make_model_stub()
    model._speech_state["req0"] = {fac_mod._GENERATE_SPEECH_KEY: True}
    tokens = torch.tensor([sampled_token])
    step = {"req0": {fac_mod._FINISH_SPEECH_KEY: True}}

    assert model.postprocess_sampled_tokens(tokens, ["req0"], {"req0": 0}, step) is tokens
    assert tokens.tolist() == [sampled_token]
    assert model._speech_state["req0"][fac_mod._GENERATE_SPEECH_KEY] is False


@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("sampled_token", [42, 99], ids=["audio-bos", "audio-eos"])
def test_pure_replay_does_not_commit_sampled_speech_transitions(active, sampled_token):
    model = _make_model_stub()
    previous = {fac_mod._GENERATE_SPEECH_KEY: active, fac_mod._FORCE_AUDIO_BOS_KEY: True}
    model._speech_state["req0"] = previous.copy()
    tokens = torch.tensor([sampled_token])
    step = {"req0": {"has_live_step": False, fac_mod._FINISH_SPEECH_KEY: True}}

    assert model.postprocess_sampled_tokens(tokens, ["req0"], {"req0": 0}, step) is tokens
    assert tokens.tolist() == [sampled_token]
    assert model._speech_state["req0"] == previous


def test_postprocess_rejects_speculative_tokens():
    model = _make_model_stub()
    with pytest.raises(ValueError, match="speculative"):
        model.postprocess_sampled_tokens(torch.tensor([[7, 8]]), ["req0"], {"req0": 0}, {})


def test_model_sampler_uses_current_sidecar_eos_before_runner_postprocess():
    model = _make_model_stub()
    model._batch_req_infos = [{"req_id": "speech"}, {"req_id": "text"}]
    model._speech_state = {
        "speech": {fac_mod._GENERATE_SPEECH_KEY: True},
        "text": {fac_mod._GENERATE_SPEECH_KEY: False},
    }
    model._batch_sidecar_results = [
        {"req_id": "speech", fac_mod._FINISH_SPEECH_KEY: True},
        {"req_id": "text", fac_mod._FINISH_SPEECH_KEY: False},
    ]
    metadata = create_autospec(SamplingMetadata, instance=True)

    def sampler(*, logits, sampling_metadata, forced_token_ids):
        assert sampling_metadata is metadata  # Text parameters are preserved.
        assert forced_token_ids == [99, None]
        return SamplerOutput(sampled_token_ids=torch.tensor([[99], [8]]), logprobs_tensors=None)

    model._ar_sampler = sampler
    output = model.sample(torch.zeros((2, 100)), metadata)
    assert output.sampled_token_ids.tolist() == [[99], [8]]
    assert model._speech_state["speech"][fac_mod._GENERATE_SPEECH_KEY] is False
    assert model._speech_state["text"][fac_mod._GENERATE_SPEECH_KEY] is False
    assert model._batch_sidecar_results[0][fac_mod._FINISH_SPEECH_KEY] is True


def test_model_sampler_forces_audio_bos_for_only_the_pending_request():
    model = _make_model_stub()
    model._batch_req_infos = [{"req_id": "pending"}, {"req_id": "text"}]
    model._batch_sidecar_results = [
        {"req_id": "pending", fac_mod._FORCE_AUDIO_BOS_KEY: True},
        {"req_id": "text", fac_mod._FORCE_AUDIO_BOS_KEY: False},
    ]

    def sampler(*, logits, sampling_metadata, forced_token_ids):
        assert forced_token_ids == [42, None]
        return SamplerOutput(sampled_token_ids=torch.tensor([[42], [8]]), logprobs_tensors=None)

    model._ar_sampler = sampler
    output = model.sample(torch.zeros((2, 100)), create_autospec(SamplingMetadata, instance=True))
    assert output.sampled_token_ids.tolist() == [[42], [8]]
    assert model._speech_state["pending"][fac_mod._GENERATE_SPEECH_KEY] is True
    assert model._speech_state["text"][fac_mod._GENERATE_SPEECH_KEY] is False


def test_chunked_initial_prefill_is_rejected_before_creating_replay_state():
    model = _make_model_stub()
    with pytest.raises(ValueError, match="complete initial prefill"):
        model.preprocess(torch.tensor([1, 2]), None, request_id="req0", _omni_prompt_len=5, _omni_num_computed_tokens=0)
    assert "req0" not in model._replay_state


def test_preprocess_single_token_text_decode_returns_text_embeddings():
    model = _make_model_stub()

    _, req_embeds, _ = model.preprocess(
        input_ids=torch.tensor([7], dtype=torch.long),
        input_embeds=None,
    )

    assert torch.equal(req_embeds, torch.zeros((1, 4), dtype=torch.float32))


def test_preprocess_first_speech_step_without_codec_history_returns_text_embeddings():
    model = _make_model_stub()

    _, req_embeds, _ = model.preprocess(
        input_ids=torch.tensor([42], dtype=torch.long),
        input_embeds=None,
        **{
            fac_mod._GENERATE_SPEECH_KEY: True,
            fac_mod._SPEECH_IDS_KEY: torch.empty((1, 0), dtype=torch.long),
        },
    )

    assert torch.equal(req_embeds, torch.zeros((1, 4), dtype=torch.float32))


def test_preprocess_active_speech_with_codec_history_blends_audio_features():
    model = _make_model_stub(hidden_size=4)

    _, req_embeds, _ = model.preprocess(
        input_ids=torch.tensor([42], dtype=torch.long),
        input_embeds=None,
        **{
            fac_mod._GENERATE_SPEECH_KEY: True,
            fac_mod._SPEECH_IDS_KEY: torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long),
        },
    )

    assert torch.equal(req_embeds, torch.full((1, 4), 1.0))


def test_preprocess_keeps_current_token_and_history_out_of_runner_cpu_buffer():
    model = _make_model_stub()
    initial_speech_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    model._speech_ids_gpu_state["req0"] = initial_speech_ids
    _, _, update = model.preprocess(
        torch.tensor([42]),
        None,
        request_id="req0",
        **{fac_mod._GENERATE_SPEECH_KEY: True, fac_mod._SPEECH_IDS_KEY: torch.full((1, 5), 99)},
    )
    assert model._speech_ids_gpu_state["req0"] is initial_speech_ids
    assert model._batch_req_infos[0][fac_mod._CURRENT_INPUT_TOKEN_ID_KEY].tolist() == [42]
    assert fac_mod._CURRENT_INPUT_TOKEN_ID_KEY not in update
    assert fac_mod._SPEECH_IDS_KEY not in update
    assert fac_mod._CRQ_PAST_KEY_VALUES_KEY not in update
    assert set(update) == {"request_id", fac_mod._TEXT_SEQ_LEN_KEY, "audio_token_ids"}


def test_audio_sidecar_persists_full_history_but_emits_only_increment():
    model = _make_model_stub(group_size=2, hidden_size=4)

    class AudioInvertTowerStub:
        def __init__(self):
            self.crq_audio_embeds = None
            self.crq_past_key_values = None
            self.crq_generate_tokens = torch.tensor([[7, 8]], dtype=torch.long)

        def crq_generate_forward(self, **kwargs):
            del kwargs

    model.audio_invert_tower = AudioInvertTowerStub()
    previous = torch.tensor([[1, 2]], dtype=torch.long)

    result = model._run_audio_sidecar_step(
        hidden_state=torch.zeros(4, dtype=torch.float32),
        current_input_token_id=torch.tensor([42], dtype=torch.long),
        speech_ids=previous,
        cached_audio_embeds=None,
        cached_past_key_values=None,
        logits_processor=[],
        do_sample=False,
        current_text_seq_len=3,
        req_id="req0",
    )

    assert torch.equal(result[fac_mod._AUDIO_TOKEN_IDS_KEY], torch.tensor([[7, 8]]))
    assert torch.equal(result[fac_mod._SPEECH_IDS_KEY], torch.tensor([[1, 2, 7, 8]]))
    assert result[fac_mod._SPEECH_IDS_KEY] is model._speech_ids_gpu_state["req0"]


def test_on_requests_finished_clears_all_resident_speech_state():
    model = _make_model_stub()
    model._crq_gpu_state["req0"] = {"embeds": object(), "pkv": object()}
    model._speech_ids_gpu_state["req0"] = torch.tensor([[1, 2]], dtype=torch.long)
    model._speech_state["req0"] = {fac_mod._GENERATE_SPEECH_KEY: True}

    model.on_requests_finished({"req0"})

    assert "req0" not in model._crq_gpu_state
    assert "req0" not in model._speech_ids_gpu_state
    assert "req0" not in model._speech_state


def test_run_audio_sidecar_decode_warmup_updates_cache_only():
    model = _make_model_stub(hidden_size=4)

    class AudioInvertTowerStub:
        def __init__(self):
            self.crq_audio_embeds = None
            self.crq_past_key_values = None
            self.crq_do_sample = None
            self.crq_logits_processor = None
            self.crq_speech_ids = None

        def crq_generate_forward(self, *, inputs_embeds, return_dict=True):
            del return_dict
            self.last_inputs_embeds = inputs_embeds
            self.crq_audio_embeds = torch.full((1, 4), 5.0, dtype=torch.float32, device=inputs_embeds.device)
            self.crq_past_key_values = (torch.full((1, 1), 7.0, dtype=torch.float32, device=inputs_embeds.device),)

    model.audio_invert_tower = AudioInvertTowerStub()

    warmup_state = model._run_audio_sidecar_decode_warmup(
        hidden_state=torch.zeros(4, dtype=torch.float32),
        current_input_token_id=7,
        speech_ids=torch.empty((1, 0), dtype=torch.long),
        cached_audio_embeds=None,
        cached_past_key_values=None,
        logits_processor=[],
        do_sample=True,
    )

    assert list(model.audio_invert_tower.last_inputs_embeds.shape) == [1, 1, 4]
    assert torch.equal(warmup_state[fac_mod._CRQ_AUDIO_EMBEDS_KEY], torch.full((1, 4), 5.0))
    assert torch.equal(warmup_state[fac_mod._CRQ_PAST_KEY_VALUES_KEY][0], torch.full((1, 1), 7.0))


def test_postprocess_prefill_warmup_updates_cache_without_emitting_audio():
    model = _make_model_stub(hidden_size=4)
    model._batch_sidecar_results = [
        {
            fac_mod._AUDIO_TOKEN_IDS_KEY: torch.full((1, 5), -1, dtype=torch.long),
            fac_mod._CRQ_AUDIO_EMBEDS_KEY: None,
            fac_mod._CRQ_PAST_KEY_VALUES_KEY: None,
            fac_mod._FORCE_AUDIO_BOS_KEY: True,
            fac_mod._FINISH_SPEECH_KEY: False,
            fac_mod._GENERATE_SPEECH_KEY: False,
            fac_mod._SPEECH_IDS_KEY: torch.empty((1, 0), dtype=torch.long),
            "end_position": 3,
            "_run_prefill_crq_warmup": True,
            "_prefill_input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
            "_prefill_crq_logits_processor": [],
            "_prefill_crq_do_sample": False,
            "audio_token_ids": torch.full((1, 5), -1, dtype=torch.long),
        }
    ]
    model._postprocess_cursor = 0

    def _prefill_warmup(**kwargs):
        del kwargs
        return {
            fac_mod._CRQ_AUDIO_EMBEDS_KEY: torch.full((1, 4), 9.0),
            fac_mod._CRQ_PAST_KEY_VALUES_KEY: (torch.full((1, 1), 3.0),),
        }

    model._run_audio_sidecar_prefill_warmup = _prefill_warmup

    output = model.postprocess(torch.zeros((3, 4), dtype=torch.float32))

    assert torch.equal(output["audio_token_ids"], torch.full((1, 5), -1, dtype=torch.long))
    assert set(output) == {"audio_token_ids"}


def _make_model_stub_for_audio_read(
    *,
    audio_token_index: int = 1000,
    group_size: int = 5,
    hidden_size: int = 4,
    num_audio_rows: int = 3,
):
    """Stub that exercises the prefill user-audio read-in path.

    The real audio towers (continuous_audio_tower / discrete audio_tower) are
    not instantiated here; instead we stub `_gather_user_audio_embeds` to
    return a fixed per-item embedding tuple and `embed_input_ids` to perform
    the realistic in-place scatter, so the test asserts that preprocess
    correctly detects the <|AUDIO|> placeholder positions and routes them
    through embed_input_ids with the right is_multimodal mask.
    """
    model = _make_model_stub(group_size=group_size, hidden_size=hidden_size)
    model.config.audio_token_index = audio_token_index

    audio_marker = 7.0
    # Fixed per-item embedding rows that `_gather_user_audio_embeds` will pretend
    # the audio towers produced. Total rows must equal the placeholder count in
    # the test's input_ids.
    fake_embeds = tuple(torch.full((1, hidden_size), audio_marker, dtype=torch.float32) for _ in range(num_audio_rows))

    def fake_gather(mm_features, device):  # noqa: ARG001
        return fake_embeds

    def fake_embed_input_ids(flat_ids, multimodal_embeddings=None, *, is_multimodal=None):
        # Mirror SupportsMultiModal.embed_input_ids scatter semantics.
        embeds = model.get_language_model().embed_input_ids(flat_ids)  # text (zeros)
        if multimodal_embeddings:
            flat = torch.cat(
                [e.reshape(-1, embeds.shape[-1]) for e in multimodal_embeddings],
                dim=0,
            )
            embeds[is_multimodal] = flat.to(dtype=embeds.dtype)
        return embeds

    model._gather_user_audio_embeds = fake_gather
    model.embed_input_ids = fake_embed_input_ids
    return model, audio_token_index, audio_marker


def test_preprocess_prefill_merges_user_audio_at_audio_token_positions():
    model, audio_tok, marker = _make_model_stub_for_audio_read(num_audio_rows=3)
    # input_ids: 2 text tokens, 3 <|AUDIO|> placeholders, 1 text token.
    input_ids = torch.tensor([1, 2, audio_tok, audio_tok, audio_tok, 3], dtype=torch.long)

    _, req_embeds, _ = model.preprocess(
        input_ids=input_ids,
        input_embeds=None,
        mm_features=[_fake_audio_mm_feature({})],  # content unused (stubbed)
    )

    # Placeholder positions overwritten with the audio marker; text positions
    # stay as the text embeddings (zeros from the stub).
    assert torch.equal(
        req_embeds[2:5],
        torch.full((3, 4), marker, dtype=torch.float32),
    )
    assert torch.equal(req_embeds[0:2], torch.zeros((2, 4), dtype=torch.float32))
    assert torch.equal(req_embeds[5], torch.zeros((4,), dtype=torch.float32))


def test_preprocess_prefill_without_mm_features_keeps_text_embeddings():
    """A text-only prefill span (no mm_features) must keep plain text embeddings."""
    model = _make_model_stub()
    model.config.audio_token_index = 1000
    # No audio_token_index tokens in the span either.
    input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)

    _, req_embeds, _ = model.preprocess(
        input_ids=input_ids,
        input_embeds=None,
    )

    assert torch.equal(req_embeds, torch.zeros((4, 4), dtype=torch.float32))


def test_preprocess_prefill_rejects_placeholder_count_mismatch():
    """A placeholder/audio-row mismatch must fail instead of dropping audio."""
    model, audio_tok, _ = _make_model_stub_for_audio_read(num_audio_rows=2)
    input_ids = torch.tensor([1, audio_tok, audio_tok, audio_tok, 2], dtype=torch.long)

    with pytest.raises(ValueError, match="placeholder count"):
        model.preprocess(
            input_ids=input_ids,
            input_embeds=None,
            mm_features=[_fake_audio_mm_feature({})],
        )


def test_preprocess_decode_span_does_not_read_user_audio():
    """Decode spans (span_len == 1) never take the user-audio read-in branch."""
    model, audio_tok, _ = _make_model_stub_for_audio_read(num_audio_rows=1)
    # A single audio_token_index token as a decode step should NOT trigger
    # _gather_user_audio_embeds (we assert via a probe).
    calls = []

    def gather_audio(mm, dev):
        calls.append(1)
        return None

    model._gather_user_audio_embeds = gather_audio

    input_ids = torch.tensor([audio_tok], dtype=torch.long)
    _, req_embeds, _ = model.preprocess(
        input_ids=input_ids,
        input_embeds=None,
        mm_features=[_fake_audio_mm_feature({})],
    )

    assert calls == []  # read-in branch not entered on decode
    # Treat the token as text -> zeros from the stub LM embedding.
    assert torch.equal(req_embeds, torch.zeros((1, 4), dtype=torch.float32))


def _fake_audio_mm_feature(per_item: dict, *, modality: str = "audio") -> MultiModalFeatureSpec:
    """Build a typed feature stub for _gather_user_audio_embeds.

    ``per_item`` maps each mm key to the per-item tensor slice as produced by
    ``MultiModalBatchedField.build_elems`` (i.e. batch dim already removed).
    Use real field elements so ``gather_kwargs`` sees native per-item data.
    """
    feature = create_autospec(MultiModalFeatureSpec, instance=True)
    feature.modality = modality
    feature.mm_position = PlaceholderRange(offset=0, length=1)
    feature.data = MultiModalKwargsItem(
        {key: MultiModalFieldElem(data=value, field=MultiModalBatchedField()) for key, value in per_item.items()}
    )
    return cast(MultiModalFeatureSpec, feature)


def test_gather_user_audio_embeds_stacks_tensor_fields_and_passes_speech_ids_as_list():
    """Regression for the on-server crash: native embed_multimodal asserts
    input_features/feature_attention_mask/feature_exist_mask are *batched
    Tensors*, while speech_ids/speech_attention_mask may be a list of 1D.
    gather_kwargs hands back per-item slices; the helper must torch.stack the
    Tensor-required fields and keep speech_ids as a list.
    """
    model = _make_model_stub(hidden_size=4)
    model.config.audio_token_index = 1000

    n_mel, n_frames, L = 80, 30, 25
    feat = {
        "speech_ids": torch.arange(L, dtype=torch.long),  # (L,)
        "speech_attention_mask": torch.ones(L, dtype=torch.long),  # (L,)
        "input_features": torch.randn(n_mel, n_frames),  # (n_mel, T)
        "feature_attention_mask": torch.ones(n_frames, dtype=torch.long),  # (T,)
        "feature_exist_mask": torch.ones(1, dtype=torch.bool),  # (1,) -> scalar item
    }
    mm_features = [_fake_audio_mm_feature(feat)]

    captured = {}

    def fake_embed_multimodal(**kwargs):
        captured.update(kwargs)
        # one audio item -> one embedding row of hidden_size
        return (torch.zeros((1, 4), dtype=torch.float32),)

    model.embed_multimodal = fake_embed_multimodal

    out = model._gather_user_audio_embeds(mm_features, device=torch.device("cpu"))
    assert out is not None and len(out) == 1

    # speech_ids must stay a list (native pads it itself).
    assert isinstance(captured["speech_ids"], list)
    assert isinstance(captured["speech_attention_mask"], list)
    # Tensor-required fields must be re-stacked batched Tensors, not lists.
    assert isinstance(captured["input_features"], torch.Tensor)
    assert captured["input_features"].shape == (1, n_mel, n_frames)
    assert isinstance(captured["feature_attention_mask"], torch.Tensor)
    assert captured["feature_attention_mask"].shape == (1, n_frames)
    assert isinstance(captured["feature_exist_mask"], torch.Tensor)
    assert captured["feature_exist_mask"].shape == (1,)


def test_gather_user_audio_embeds_returns_none_when_no_audio_or_no_speech_ids():
    model = _make_model_stub()
    model.config.audio_token_index = 1000
    model.embed_multimodal = lambda **k: None  # should not be called

    # Empty / non-audio features.
    assert model._gather_user_audio_embeds([], torch.device("cpu")) is None
    assert (
        model._gather_user_audio_embeds(
            [_fake_audio_mm_feature({}, modality="image")],
            torch.device("cpu"),
        )
        is None
    )
    # Audio feature without speech_ids.
    assert (
        model._gather_user_audio_embeds(
            [_fake_audio_mm_feature({"input_features": torch.randn(80, 30)})],
            torch.device("cpu"),
        )
        is None
    )


def test_crq_sampling_seed_is_isolated_across_interleaved_requests():
    model = _make_model_stub()
    model.audio_invert_tower = SimpleNamespace()
    model._patch_audio_invert_tower_sampling_step()
    tower = model.audio_invert_tower
    tower.crq_logits_processor = lambda ids, logits: logits
    tower.crq_speech_ids = torch.empty(1, 0, dtype=torch.long)
    tower.crq_generate_tokens = []
    tower.crq_do_sample = True
    for rid in ("alone", "interleaved", "other"):
        model.preprocess(torch.tensor([1, 2]), None, request_id=rid, _omni_seed=42)
    logits = torch.zeros(1, 1, 100)

    def sample(rid):
        tower.crq_generator = model._crq_generators[rid]
        return tower.sampling_step(logits)[0].item()

    alone = [sample("alone") for _ in range(20)]
    interleaved = []
    for _ in range(20):
        sample("other")
        torch.rand(13)  # Unrelated global RNG use must not affect the request.
        interleaved.append(sample("interleaved"))
    assert alone == interleaved
    generator = model._crq_generators["interleaved"]
    model._replay_state["interleaved"].processed_until = 2
    model.preprocess(torch.tensor([3]), None, request_id="interleaved", _omni_seed=42)
    assert model._crq_generators["interleaved"] is generator
    model.on_requests_finished({"interleaved"})
    assert "interleaved" not in model._crq_generators
    assert "other" in model._crq_generators
