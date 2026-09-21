# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from pathlib import Path

import pytest

from vllm_omni.config.pipeline_registry import resolve_pipeline_config
from vllm_omni.config.stage_config import load_deploy_config, merge_pipeline_deploy

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("async_chunk", [False, True])
def test_funaudiochat_deploy_selects_matching_handoff(async_chunk):
    pipeline = resolve_pipeline_config("funaudiochat")
    assert pipeline is not None
    path = Path(__file__).resolve().parents[2] / "vllm_omni/deploy/funaudiochat.yaml"
    deploy = load_deploy_config(path)
    deploy.async_chunk = async_chunk
    stages = merge_pipeline_deploy(pipeline, deploy)

    assert len(stages) == 2
    assert stages[0].is_comprehension
    assert stages[0].final_output_type == "text"
    assert stages[1].input_sources == [0]
    assert stages[1].final_output_type == "audio"
    assert stages[1].yaml_engine_args["model"] == "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
    assert stages[1].sampling_constraints["detokenize"] is False
    if async_chunk:
        assert (
            stages[0]
            .yaml_engine_args["custom_process_next_stage_input_func"]
            .endswith(".funaudiochat2code2wav_async_chunk")
        )
    else:
        assert stages[1].custom_process_input_func.endswith(".funaudiochat2code2wav")


def test_funaudiochat_does_not_advertise_continuous_input():
    pipeline = resolve_pipeline_config("funaudiochat")
    assert pipeline.duplex_plugin is None
