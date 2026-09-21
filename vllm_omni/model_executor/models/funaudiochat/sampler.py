# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import torch
from vllm.config.model import LogprobsMode
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler


class FunAudioChatSampler(Sampler):
    """Apply mandatory speech boundaries before token/logprob selection.

    Boundary tokens are part of the model's speech protocol and take
    precedence over ordinary user sampling restrictions. Applying the mask
    after the standard processors prevents min_tokens or allowed_token_ids
    from turning a forced row into an all-negative-infinity distribution.
    """

    _forced_token_ids: list[int | None] | None = None

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        predict_bonus_token: bool = False,
        logprobs_mode_override: LogprobsMode | None = None,
        *,
        forced_token_ids: list[int | None] | None = None,
    ) -> SamplerOutput:
        if forced_token_ids is not None and len(forced_token_ids) != logits.shape[0]:
            raise ValueError("FunAudioChat forced-token rows do not match the sampling batch")
        self._forced_token_ids = forced_token_ids
        try:
            return super().forward(logits, sampling_metadata, predict_bonus_token, logprobs_mode_override)
        finally:
            self._forced_token_ids = None

    def apply_logits_processors(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        predict_bonus_token: bool,
    ) -> torch.Tensor:
        logits = super().apply_logits_processors(logits, sampling_metadata, predict_bonus_token)
        forced = self._forced_token_ids
        if forced is not None:
            for row, token_id in enumerate(forced):
                if token_id is not None:
                    logits[row].fill_(float("-inf"))
                    logits[row, token_id] = 0.0
        return logits
