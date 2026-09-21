# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import threading
from collections import defaultdict
from types import MethodType, SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav import CosyVoice3Code2Wav
from vllm_omni.model_executor.models.funaudiochat.funaudiochat_code2wav import (
    FunAudioChatCosyVoice3Code2Wav,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.stage_input_processors.funaudiochat import funaudiochat2code2wav_async_chunk

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_split_tokens_like_official_keeps_short_inputs_as_single_segment():
    token = torch.arange(100, dtype=torch.long)

    segments = FunAudioChatCosyVoice3Code2Wav._split_tokens_like_official(token)

    assert len(segments) == 1
    assert torch.equal(segments[0], token)


def test_split_tokens_like_official_rebalances_tiny_tail_segment():
    token = torch.arange(760, dtype=torch.long)

    segments = FunAudioChatCosyVoice3Code2Wav._split_tokens_like_official(token)

    assert [segment.numel() for segment in segments] == [380, 380]
    assert torch.equal(torch.cat(segments, dim=0), token)


def _build_code2wav_stub() -> FunAudioChatCosyVoice3Code2Wav:
    model = object.__new__(FunAudioChatCosyVoice3Code2Wav)
    model.vllm_config = SimpleNamespace(device_config=SimpleNamespace(device=torch.device("cpu")))
    model._max_codec_token_id = 6560
    model._dummy_profile_token_len = 32
    model._logged_dummy_profile_cap = False
    return model


def test_build_decode_tokens_keeps_real_input_ids_without_sampling_metadata():
    model = _build_code2wav_stub()
    input_ids = torch.tensor([12, 34, 56], dtype=torch.long)

    token_batches, is_dummy_profile = model._build_decode_tokens(input_ids, sampling_metadata=None)

    assert len(token_batches) == 1
    assert token_batches[0].tolist() == [[12, 34, 56]]
    assert is_dummy_profile is False


def test_build_decode_tokens_uses_prompt_token_ids_when_input_ids_are_empty():
    model = _build_code2wav_stub()
    sampling_metadata = SimpleNamespace(prompt_token_ids=[1, 2, 3, 4])

    token_batches, is_dummy_profile = model._build_decode_tokens(
        torch.empty((0,), dtype=torch.long),
        sampling_metadata,
    )

    assert len(token_batches) == 1
    assert token_batches[0].tolist() == [[1, 2, 3, 4]]
    assert is_dummy_profile is False


def test_build_decode_tokens_treats_all_zero_missing_metadata_as_dummy_profile():
    model = _build_code2wav_stub()
    input_ids = torch.zeros((64,), dtype=torch.long)

    token_batches, is_dummy_profile = model._build_decode_tokens(input_ids, sampling_metadata=None)

    assert len(token_batches) == 1
    assert token_batches[0].shape == (1, 32)
    assert is_dummy_profile is True


def test_build_decode_tokens_no_longer_rejects_long_sequences_before_segmentation():
    model = _build_code2wav_stub()
    input_ids = torch.arange(10235, dtype=torch.long) % 6000

    token_batches, is_dummy_profile = model._build_decode_tokens(input_ids, sampling_metadata=None)

    assert len(token_batches) == 1
    assert token_batches[0].shape == (1, 10235)
    assert is_dummy_profile is False


def test_build_decode_tokens_preserves_batched_prompt_token_ids_per_request():
    model = _build_code2wav_stub()
    sampling_metadata = SimpleNamespace(prompt_token_ids=[[1, 2, 3], [4, 5]])

    token_batches, is_dummy_profile = model._build_decode_tokens(
        torch.empty((0,), dtype=torch.long),
        sampling_metadata,
    )

    assert [token.tolist() for token in token_batches] == [[[1, 2, 3]], [[4, 5]]]
    assert is_dummy_profile is False


# --- streaming forward (async_chunk path) -------------------------------------


def _build_streaming_stub() -> FunAudioChatCosyVoice3Code2Wav:
    """Build a stub whose ``forward`` exercises the streaming dispatch branch.

    Bounded Flow and fixed-cache HiFT are replaced with recorders so offsets,
    cache carry, and finalization are observable without loading real weights.
    """
    model = _build_code2wav_stub()
    model._max_codec_token_id = 6560
    model._code2wav_sample_rate = 24000
    model._stream_vocoder_cache_by_req = {}
    model._stream_audio_cache_lock = threading.Lock()
    model._speaker_embedding = torch.zeros((1, 80), dtype=torch.float32)

    calls: list[dict] = []

    def fake_flow_segment(token, *, token_offset, finalize):
        calls.append(
            {
                "num_tokens": int(token.numel()),
                "offset": int(token_offset),
                "finalize": bool(finalize),
            }
        )
        return torch.full((1, 80, 6), 0.25, dtype=torch.float32)

    def fake_hift_chunk(chunk_mel, cache_state, *, finalize):
        calls[-1]["had_cache"] = cache_state is not None
        calls[-1]["mel_len"] = int(chunk_mel.shape[-1])
        suffix = torch.full(
            (
                1,
                1,
                100,
            ),
            0.125,
            dtype=torch.float32,
        )
        new_state = None if finalize else {"_call": len(calls)}
        return suffix, new_state

    model.code2wav = SimpleNamespace(output_size=80)
    model._run_streaming_flow_segment = fake_flow_segment
    model._run_streaming_hift_chunk = fake_hift_chunk
    model._forward_calls = calls
    return model


def _chunk_meta(
    left_context_size,
    *,
    stream_finished=False,
    req_id="r1",
    segment_start=0,
):
    return {
        "meta": {
            "finished": torch.tensor(stream_finished, dtype=torch.bool),
            "stream_finished": torch.tensor(stream_finished, dtype=torch.bool)
            if stream_finished
            else torch.tensor(False, dtype=torch.bool),
            "req_id": [req_id],
            "left_context_size": left_context_size,
            "num_processed_tokens": segment_start,
        },
        "codes": {"audio": torch.tensor([0], dtype=torch.long)},
    }


def test_forward_streams_incrementally_and_carries_cache_across_chunks():
    model = _build_streaming_stub()

    # First chunk: offset 0, not finalize. Cache should be None on entry.
    out1 = model.forward(
        input_ids=torch.tensor([1, 2, 3], dtype=torch.long),
        sampling_metadata=None,
        seq_token_counts=[3],
        model_intermediate_buffer=[_chunk_meta(0)],
    )
    call1 = model._forward_calls[-1]
    assert call1["offset"] == 0
    assert call1["finalize"] is False
    assert call1["had_cache"] is False
    assert isinstance(out1, OmniOutput)
    assert out1.multimodal_outputs["audio"][0].numel() == 100
    # The carried cache must now live under the request id.
    assert "r1" in model._stream_vocoder_cache_by_req

    # Second chunk: cumulative prefix offset and carried cache are forwarded.
    model.forward(
        input_ids=torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
        sampling_metadata=None,
        seq_token_counts=[5],
        model_intermediate_buffer=[_chunk_meta(2)],
    )
    call2 = model._forward_calls[-1]
    assert call2["offset"] == 2
    assert call2["had_cache"] is True

    # Final chunk: stream_finished clears the per-request cache.
    model.forward(
        input_ids=torch.tensor([1, 2, 3], dtype=torch.long),
        sampling_metadata=None,
        seq_token_counts=[3],
        model_intermediate_buffer=[_chunk_meta(2, stream_finished=True)],
    )
    call3 = model._forward_calls[-1]
    assert call3["finalize"] is True
    assert "r1" not in model._stream_vocoder_cache_by_req


def test_on_requests_finished_releases_only_completed_request_cache():
    model = _build_streaming_stub()
    model._stream_vocoder_cache_by_req = {"r1": {"cache": 1}, "r2": {"cache": 2}}

    model.on_requests_finished(["r1"])

    assert "r1" not in model._stream_vocoder_cache_by_req
    assert model._stream_vocoder_cache_by_req["r2"] == {"cache": 2}


def test_forward_empty_terminal_chunk_flushes_fixed_hift_state():
    model = _build_streaming_stub()
    model.forward(
        input_ids=torch.tensor([1, 2, 3], dtype=torch.long),
        sampling_metadata=None,
        seq_token_counts=[3],
        model_intermediate_buffer=[_chunk_meta(0)],
    )
    assert "r1" in model._stream_vocoder_cache_by_req
    flow_call_count = len(model._forward_calls)

    out = model.forward(
        input_ids=torch.empty((0,), dtype=torch.long),
        sampling_metadata=None,
        seq_token_counts=[0],
        model_intermediate_buffer=[_chunk_meta(0, stream_finished=True)],
    )

    # Empty terminal input skips Flow but still flushes the fixed HiFT cache.
    assert len(model._forward_calls) == flow_call_count
    assert out.multimodal_outputs["audio"][0].numel() == 100
    assert "r1" not in model._stream_vocoder_cache_by_req


def test_forward_falls_back_to_non_streaming_without_intermediate_buffer():
    """When no chunk metadata is present, the full-segment path runs unchanged.

    We can't exercise ``_decode_segment_like_official`` without weights, but we can
    assert the streaming dispatch is skipped (no bounded Flow call recorded)
    and the branch selection is correct. With a real model the decode executes below.
    """
    model = _build_streaming_stub()
    # Stub out the heavy non-streaming decode so this test stays unit-level.
    captured: list[torch.Tensor] = []

    def fake_decode_segment(token_segment, prompt_token, prompt_feat, embedding):
        captured.append(token_segment)
        return torch.ones(token_segment.numel(), dtype=torch.float32) * 0.5

    model._decode_segment_like_official = fake_decode_segment
    # ``_split_tokens_like_official`` is a staticmethod on the real class; reuse its
    # underlying function so the instance-bound dispatch in ``forward`` still works.
    model._split_tokens_like_official = FunAudioChatCosyVoice3Code2Wav._split_tokens_like_official

    out = model.forward(
        input_ids=torch.arange(80, dtype=torch.long),
        sampling_metadata=None,
        seq_token_counts=[80],
        model_intermediate_buffer=[{}],  # no meta -> non-streaming branch
    )
    assert model._forward_calls == []  # streaming flow never invoked
    assert captured  # the non-streaming segment decode ran
    assert out.multimodal_outputs["audio"][0].numel() > 0


def test_forward_handles_empty_runtime_info_defensively():
    """An empty buffer list (e.g. dummy/profile) must not raise from to_struct."""
    model = _build_streaming_stub()
    # No model_intermediate_buffer kwarg at all — still must reach the empty early return.
    out = model.forward(input_ids=torch.zeros((0,), dtype=torch.long), sampling_metadata=None)
    assert out.multimodal_outputs["audio"][0].numel() == 0
    assert model._forward_calls == []


def test_init_reuses_resolved_config_after_temporary_file_is_removed(mocker, tmp_path):
    from vllm_omni.model_executor.models.funaudiochat import funaudiochat_code2wav as module
    from vllm_omni.transformers_utils.configs.cosyvoice3 import CosyVoice3Config

    config = CosyVoice3Config()
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            model=str(tmp_path),
            hf_config=config,
            hf_config_path=str(tmp_path / "removed-config"),
        ),
        device_config=SimpleNamespace(device=torch.device("cpu")),
    )
    codec_factory = mocker.patch.object(module, "CosyVoice3Code2Wav")
    codec_factory.return_value.mel_cache_len = 8
    attention = mocker.Mock(spec=module.Attention)
    attention.allow_fp32_fallback = False
    codec_factory.return_value.flow_model.float.return_value.modules.return_value = [attention]
    mocker.patch.object(
        module.FunAudioChatCosyVoice3Code2Wav, "_load_default_speaker_embedding", return_value=torch.zeros(1, 192)
    )
    mocker.patch.object(module.FunAudioChatCosyVoice3Code2Wav, "_compute_max_supported_token_len", return_value=7500)
    model = module.FunAudioChatCosyVoice3Code2Wav(vllm_config=vllm_config)
    assert model.config is config
    codec_factory.assert_called_once_with(config)
    assert attention.allow_fp32_fallback is True


def test_hift_accepts_source_module_phase_result(mocker):
    model = _build_code2wav_stub()
    hift = mocker.Mock()
    mel = torch.ones(1, 80, 4)
    hift.f0_predictor.return_value = torch.ones(1, 4)
    hift.f0_upsamp.return_value = torch.ones(1, 1, 4)
    hift.m_source.return_value = (torch.ones(1, 4, 1), None, None, torch.zeros(1))
    hift.decode.return_value = torch.ones(1, 4)
    model.code2wav = SimpleNamespace(hift=hift)
    speech, source = model._run_hift_like_official(mel, finalize=True, cache_source=torch.full((1, 1, 2), 7.0))
    assert speech.shape == (1, 4)
    assert source.tolist() == [[[7.0, 7.0, 1.0, 1.0]]]
    hift.decode.assert_called_once()


def _build_shared_hift_stub():
    """Exercise the real shared window/emission logic with an indexed waveform.

    Only the neural Flow/HiFT calls are replaced. The HiFT double models its
    actual 3 F0 + 4 convolution + 1 ISTFT lookahead frames and returns distinct
    phase markers, so missing samples, duplicated overlap, and lost phase
    become observable through the FunAudioChat request-cache path.
    """
    model = _build_code2wav_stub()
    model._stream_audio_cache_lock = threading.Lock()
    model._stream_vocoder_cache_by_req = {}
    calls = []

    def inference(*, speech_feat, finalize, phase_acc, uv_offset, next_overlap, trim, f0_margin):
        calls.append(
            {
                "phase_acc": phase_acc,
                "uv_offset": uv_offset,
                "trim": trim,
                "finalize": finalize,
                "f0_margin": f0_margin,
            }
        )
        withheld_frames = 0 if finalize else 8
        sample_count = (speech_feat.shape[-1] - f0_margin - withheld_frames) * 480
        speech = torch.arange(uv_offset, uv_offset + sample_count, dtype=torch.float32).reshape(1, 1, -1)
        return speech, torch.empty(1, 1, 0), torch.tensor([float(len(calls))])

    hift = SimpleNamespace(
        m_source=SimpleNamespace(l_linear=SimpleNamespace(weight=torch.zeros(1))),
        upsample_rates=[8, 5, 3],
        istft_params={"hop_len": 4},
        f0_predictor=SimpleNamespace(condnet=[SimpleNamespace(causal_padding=3)], left_context_frames=8),
        conv_pre_look_right=4,
        inference=inference,
    )
    codec = SimpleNamespace(hift=hift, _hift_window_len=64, output_size=80)
    codec._stream_hift_from_feat = MethodType(CosyVoice3Code2Wav._stream_hift_from_feat, codec)
    model.code2wav = codec

    def flow_segment(token, *, token_offset, finalize):
        token_count = token.numel() - token_offset - (0 if finalize else 3)
        return torch.ones(1, 80, token_count * 2)

    model._run_streaming_flow_segment = flow_segment
    return model, calls


def test_streaming_fixed_codec_sequence_preserves_all_samples_and_phase():
    """Regression for 415 codec tokens losing 3,840 samples at finalization."""
    model, calls = _build_shared_hift_stub()
    manager = SimpleNamespace(request_payload=defaultdict(dict), code_prompt_token_ids=defaultdict(list))
    request = SimpleNamespace(request_id="fixed-codes", is_finished=lambda: False)
    codec_ids = torch.arange(415)
    pieces = []
    for index in range(0, codec_ids.numel(), 5):
        payload = funaudiochat2code2wav_async_chunk(
            manager,
            {"audio_token_ids": codec_ids[index : index + 5]},
            request,
            is_finished=index + 5 == codec_ids.numel(),
        )
        if payload is None:
            continue
        pieces.append(
            model._decode_request_streaming(
                payload.codes.audio,
                request.request_id,
                payload.meta,
                finalize=bool(payload.meta.stream_finished),
            )
        )

    waveform = torch.cat(pieces)
    assert pieces[0].numel() == 5760  # Ten codec frames minus the eight-mel lookahead.
    assert waveform.numel() == 398400  # 415 * 2 mel frames * 480 samples.
    torch.testing.assert_close(waveform, torch.arange(398400, dtype=torch.float32), rtol=0, atol=0)
    assert calls[0]["phase_acc"] is None
    for index, call in enumerate(calls[1:], start=1):
        assert call["phase_acc"].item() == index
    assert any(call["uv_offset"] > 0 and call["f0_margin"] > 0 for call in calls)
    assert calls[-1]["finalize"] is True and calls[-1]["trim"] == 0
    assert model._stream_vocoder_cache_by_req == {}


def test_empty_final_chunk_releases_shared_hift_tail_and_preserves_other_requests():
    model, calls = _build_shared_hift_stub()
    first = model._decode_request_streaming(
        torch.arange(13), "r1", SimpleNamespace(left_context_size=0), finalize=False
    )
    model._stream_vocoder_cache_by_req["other"] = {"sentinel": True}
    tail = model._decode_request_streaming(
        torch.empty(0, dtype=torch.long), "r1", SimpleNamespace(left_context_size=0), finalize=True
    )

    assert first.numel() == 5760
    assert tail.numel() == 3840
    torch.testing.assert_close(torch.cat([first, tail]), torch.arange(9600, dtype=torch.float32), rtol=0, atol=0)
    assert calls[-1]["phase_acc"].item() == 1
    assert calls[-1]["finalize"] is True
    assert model._stream_vocoder_cache_by_req == {"other": {"sentinel": True}}
