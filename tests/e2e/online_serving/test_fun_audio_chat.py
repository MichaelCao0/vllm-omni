# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Complete-turn audio input and streaming speech output for Fun-Audio-Chat."""

import os
import runpy
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import numpy as np
import pybase64
import pytest
import soundfile as sf

from tests.helpers.client import OmniResponse
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.model_executor.models.funaudiochat.common import resolve_funaudiochat_root

_MODEL = os.environ.get("FUNAUDIOCHAT_MODEL", "FunAudioLLM/Fun-Audio-Chat-8B")
_DEPLOY = os.environ.get("FUNAUDIOCHAT_DEPLOY_CONFIG", get_deploy_config_path("funaudiochat.yaml"))
_PARAMS = [
    pytest.param(
        OmniServerParams(
            model=_MODEL,
            stage_config_path=_DEPLOY,
        ),
        id="funaudiochat",
    )
]
_AUDIO_KEYWORDS = ["music", "piano", "ambient", "relax", "sleep", "rain", "noise", "slow"]
_MAX_TOKENS = 512


def _speech_request(model, *, stream, text=None, keywords=None):
    # Load the pinned reference's spoken persona without importing a generic
    # top-level `utils` package or resolving the optional source at collection.
    root = resolve_funaudiochat_root()
    system_prompt = runpy.run_path(str(root / "utils" / "constant.py"))["SPOKEN_S2M_PROMPT"]
    if text is None:
        sample = root / "examples" / "ck7vv9ag.wav"
        content = [
            {
                "type": "input_audio",
                "input_audio": {"data": pybase64.b64encode(sample.read_bytes()).decode(), "format": "wav"},
            }
        ]
    else:
        content = text
    request = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "modalities": ["text", "audio"],
        "stream": stream,
        "extra_body": {"max_tokens": _MAX_TOKENS, "temperature": 0.0, "seed": 42},
        # The shared advanced_model assertion also compares the Whisper
        # transcript with the generated text; keywords check input relevance.
        "key_words": {"text": _AUDIO_KEYWORDS if keywords is None else keywords},
        "transcript_language": "en",
    }
    if stream:
        request["stream_options"] = {"include_usage": True}
    return request


def _assert_complete_response(response: OmniResponse, *, stream: bool) -> None:
    # core_model's shared assertion checks transport success only. Explicitly
    # require usable model output at both levels, before claiming an S2S pass.
    # Full multimodal responses have a final choice for each modality; SSE
    # merges both stages into one stream with one terminal choice.
    expected_finishes = ["stop"] if stream else ["stop", "stop"]
    assert response.finish_reasons == expected_finishes, (
        f"Expected natural completion per output, got finish reasons {response.finish_reasons}"
    )
    # The final audio modality can report stop after a truncated text stage.
    # Require final text-stage usage below the generation cap as well.
    assert response.completion_tokens is not None, "No final completion-token usage"
    assert 0 < response.completion_tokens < _MAX_TOKENS, (
        f"Generation did not finish below its token budget: {response.completion_tokens}/{_MAX_TOKENS}"
    )
    assert response.text_content and response.text_content.strip(), "No text output"
    assert response.audio_bytes, "No audio output"
    audio, sample_rate = sf.read(BytesIO(response.audio_bytes), dtype="float32", always_2d=True)
    assert sample_rate == 24000
    assert audio.shape[1] == 1, f"Expected mono speech, got {audio.shape}"
    assert audio.shape[0] >= sample_rate // 2, "Generated speech is shorter than 0.5 seconds"
    assert np.isfinite(audio).all(), "Generated audio contains NaN or infinity"
    assert np.max(np.abs(audio)) > 1e-5, "Generated audio is silent"
    if stream:
        valid_chunks = 0
        for encoded in response.audio_data or []:
            chunk, chunk_rate = sf.read(BytesIO(pybase64.b64decode(encoded.split(",", 1)[-1])), dtype="float32")
            assert chunk_rate == sample_rate
            assert np.isfinite(chunk).all(), "Audio chunk contains NaN or infinity"
            valid_chunks += chunk.size > 0
        assert valid_chunks >= 2, "Streaming response did not contain multiple audio chunks"


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", _PARAMS, indirect=True)
def test_speech_to_text_audio(omni_server, online_client):
    responses = online_client.send_omni_request(_speech_request(omni_server.model, stream=False))
    assert len(responses) == 1
    _assert_complete_response(responses[0], stream=False)


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", _PARAMS, indirect=True)
def test_speech_to_streaming_text_audio(omni_server, online_client):
    responses = online_client.send_omni_request(_speech_request(omni_server.model, stream=True))
    assert len(responses) == 1
    _assert_complete_response(responses[0], stream=True)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", _PARAMS, indirect=True)
def test_mixed_concurrent_text_and_speech_inputs(omni_server, online_client):
    requests = [
        _speech_request(omni_server.model, stream=True),
        _speech_request(
            omni_server.model,
            stream=False,
            text="What is the capital of France? Answer in one short English sentence.",
            keywords=["paris"],
        ),
        _speech_request(
            omni_server.model,
            stream=True,
            text="What is the capital of Japan? Answer in one short English sentence.",
            keywords=["tokyo"],
        ),
        _speech_request(
            omni_server.model,
            stream=False,
            text="What is two plus two? Start your English answer with 'Two plus two equals'.",
            keywords=["four", "4"],
        ),
    ]
    # The shared client has no heterogeneous-batch API. Keep its response and
    # content assertions while submitting four distinct requests concurrently.
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(online_client.send_omni_request, config) for config in requests]
        for config, future in zip(requests, futures, strict=True):
            responses = future.result()
            assert len(responses) == 1
            _assert_complete_response(responses[0], stream=config["stream"])
