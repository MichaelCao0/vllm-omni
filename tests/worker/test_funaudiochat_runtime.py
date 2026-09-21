# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_preprocess_opt_in_attaches_audio_features_without_mutating_buffer():
    runner = SimpleNamespace(model=SimpleNamespace(wants_mm_features_in_preprocess=True))
    features = [object()]
    original = {"other": 1}
    result = OmniGPUModelRunner._maybe_attach_mimo_audio_req_infos(
        runner, SimpleNamespace(mm_features=features), original, "request-a"
    )
    assert result["mm_features"] is features
    assert result["req_id"] == "request-a"
    assert original == {"other": 1}


def test_sidecar_payload_contains_only_requested_delta_for_selected_request():
    runner = object.__new__(GPUARModelRunner)
    runner.model = SimpleNamespace(pooler_output_buffer_keys=("audio_token_ids",))
    runner.model_intermediate_buffer = {
        "request-a": {"audio_token_ids": torch.tensor([[0, 4]]), "private_kv": torch.ones(8)},
        "request-b": {"audio_token_ids": torch.tensor([[5, 6]])},
    }
    result = runner._build_omni_pooler_payload(
        rid="request-a",
        idx=0,
        start=0,
        end=1,
        hidden_states_cpu=None,
        req_hidden_states_cpu=None,
        combined_hidden_states=None,
        combined_multimodal_outputs=None,
        mm_cpu=None,
        audio_sparse_output=False,
        sparse_mm_index={},
        hidden_seq_len=1,
        scheduled_seq_len=1,
    )
    assert set(result) == {"audio_token_ids"}
    assert result["audio_token_ids"].tolist() == [[0, 4]]
