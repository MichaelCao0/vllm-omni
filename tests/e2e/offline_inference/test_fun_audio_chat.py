# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Offline complete-turn speech input and complete text/audio output."""

import os
import runpy
from io import BytesIO

import pytest
import soundfile as sf
import torch

from tests.helpers.assertions import assert_omni_response
from tests.helpers.client import OmniResponse
from tests.helpers.mark import hardware_test
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.model_executor.models.funaudiochat.common import resolve_funaudiochat_root

_MODEL = os.environ.get("FUNAUDIOCHAT_MODEL", "FunAudioLLM/Fun-Audio-Chat-8B")
_DEPLOY = os.environ.get("FUNAUDIOCHAT_DEPLOY_CONFIG", get_deploy_config_path("funaudiochat.yaml"))
_PARAMS = [(_MODEL, _DEPLOY, {"trust_remote_code": True})]


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", _PARAMS, indirect=True)
def test_speech_to_complete_text_audio(omni_runner, run_level):
    root = resolve_funaudiochat_root()
    # Match the official spoken reference without changing Python's import path.
    system_prompt = runpy.run_path(str(root / "utils" / "constant.py"))["SPOKEN_S2M_PROMPT"]
    sample = root / "examples" / "ck7vv9ag.wav"
    audio, input_sample_rate = sf.read(sample, dtype="float32")
    sampling_params = [params.clone() for params in omni_runner.get_default_sampling_params_list()]
    sampling_params[0].max_tokens = 512
    # Reuse the standard lifecycle and multimodal prompt builder, explicitly
    # supplying FunAudioChat's spoken S2M persona instead of Qwen's default.
    outputs = omni_runner.generate_multimodal(
        prompts="",
        system_prompt=system_prompt,
        audios=(audio, input_sample_rate),
        modalities=["text", "audio"],
        sampling_params_list=sampling_params,
    )
    text_outputs = [output for output in outputs if output.final_output_type == "text"]
    audio_outputs = [output for output in outputs if output.final_output_type == "audio"]
    assert len(text_outputs) == len(audio_outputs) == 1, "Expected both final stage outputs"
    assert text_outputs[0].finished and audio_outputs[0].finished, "Pipeline returned incomplete output"
    assert text_outputs[0].outputs[0].finish_reason == "stop", "Speech generation reached the token limit"
    text = text_outputs[0].outputs[0].text
    assert text and text.strip(), "No text output"
    multimodal_output = audio_outputs[0].outputs[0].multimodal_output
    raw_audio = multimodal_output.get("audio")
    assert raw_audio is not None, "No audio output"
    chunks = raw_audio if isinstance(raw_audio, list) else [raw_audio]
    assert chunks, "No accumulated audio chunks"
    waveform = torch.cat([torch.as_tensor(chunk).detach().float().cpu().reshape(-1) for chunk in chunks])
    assert waveform.numel() >= 12000, "Generated speech is shorter than 0.5 seconds"
    assert torch.isfinite(waveform).all(), "Generated audio contains NaN or infinity"
    assert waveform.abs().max() > 1e-5, "Generated audio is silent"
    raw_sample_rate = multimodal_output.get("sr")
    assert raw_sample_rate is not None, "Output did not report its sample rate"
    sample_rates = raw_sample_rate if isinstance(raw_sample_rate, list) else [raw_sample_rate]
    assert sample_rates and all(torch.as_tensor(rate).eq(24000).all() for rate in sample_rates)

    wav = BytesIO()
    sf.write(wav, waveform.numpy(), 24000, format="WAV", subtype="PCM_16")
    response = OmniResponse(success=True, text_content=text, audio_bytes=wav.getvalue())
    assert_omni_response(
        response,
        {
            "modalities": ["text", "audio"],
            "key_words": {"text": ["music", "piano", "ambient", "relax", "sleep", "rain", "noise", "slow"]},
            "transcript_language": "en",
        },
        run_level=run_level,
    )
