# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from types import MethodType
from typing import Any

import torch
import torch.nn as nn
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopPLogitsWarper,
)
from transformers.modeling_outputs import BaseModelOutput
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_omni.model_executor.models.funaudiochat.common import (
    ensure_funaudiochat_importable,
    register_funaudiochat_processor,
)
from vllm_omni.model_executor.models.funaudiochat.sampler import FunAudioChatSampler

try:
    from vllm.model_executor.models.funaudiochat import (
        FunAudioChatForConditionalGeneration as VllmNativeFunAudioChatForConditionalGeneration,
    )
except ImportError:  # pragma: no cover - environment-specific dependency
    VllmNativeFunAudioChatForConditionalGeneration = None

_NativeFunAudioChatBase = (
    VllmNativeFunAudioChatForConditionalGeneration
    if VllmNativeFunAudioChatForConditionalGeneration is not None
    else nn.Module
)

logger = init_logger(__name__)

DEFAULT_SP_GEN_KWARGS = {
    "text_greedy": True,
    "only_crq_sampling": True,
    "disable_speech": False,
    "force_text_abos": True,
}

_OFFICIAL_CRQ_SAMPLING_DEFAULTS = {
    "repetition_penalty": 1.2,
    "temperature": 0.8,
    "top_p": 0.9,
    "top_k": 0,
}

_AUDIO_TOKEN_IDS_KEY = "funaudiochat_audio_token_ids"
_CRQ_AUDIO_EMBEDS_KEY = "funaudiochat_crq_audio_embeds"
_CRQ_PAST_KEY_VALUES_KEY = "funaudiochat_crq_past_key_values"
_CURRENT_INPUT_TOKEN_ID_KEY = "funaudiochat_current_input_token_id"
_FORCE_AUDIO_BOS_KEY = "funaudiochat_force_audio_bos_pending"
_FINISH_SPEECH_KEY = "funaudiochat_finish_speech"
_GENERATE_SPEECH_KEY = "funaudiochat_generate_speech"
_SPEECH_IDS_KEY = "funaudiochat_speech_ids"
_TEXT_INPUT_IDS_KEY = "funaudiochat_text_input_ids"
_TEXT_SEQ_LEN_KEY = "funaudiochat_text_seq_len"


@dataclass
class _ReplayState:
    """Rebuild paged LLM KV without rewinding the resident CRQ sidecar."""

    prompt_len: int
    prompt_embeds: torch.Tensor
    speech_enabled: bool = True
    processed_until: int = 0
    replay_logged: bool = False
    # Input feedback actually used at each generated position. None means
    # plain text; a tensor holds one codec group, never cumulative history.
    feedback: dict[int, torch.Tensor | None] = field(default_factory=dict)


@register_funaudiochat_processor
class FunAudioChatForConditionalGeneration(_NativeFunAudioChatBase, SupportsMultiModal):
    supports_multimodal_raw_input_only = True
    supports_multimodal = True
    requires_raw_input_tokens = False
    input_modalities = "audio"
    pooler_output_buffer_keys = ("audio_token_ids",)
    # Ask the omni runner to forward the request's mm_features into the
    # per-request preprocess info_dict, so the prefill span can read the
    # user-uploaded audio (see _gather_user_audio_embeds / preprocess).
    wants_mm_features_in_preprocess = True
    prefer_model_sampler = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        if VllmNativeFunAudioChatForConditionalGeneration is None:
            raise ImportError(
                "Installed vLLM does not expose a native FunAudioChat model. "
                "Upgrade vLLM to a build that includes "
                "`vllm.model_executor.models.funaudiochat`."
            )

        super().__init__(vllm_config=vllm_config, prefix=prefix)
        ensure_funaudiochat_importable()
        from funaudiochat.modeling_funaudiochat import FunAudioChatDecoder  # type: ignore

        self.audio_invert_tower = FunAudioChatDecoder(self.config.audio_config)
        self._patch_audio_invert_tower_sampling_step()
        self.sp_gen_kwargs = DEFAULT_SP_GEN_KWARGS.copy()
        self.has_preprocess = True
        self.has_postprocess = True
        # CRQ postprocess consumes live prefill hidden states and emits codec
        # deltas through the request buffer. Keep output construction synchronous
        # until those deltas have a step-owned snapshot for the async builder.
        self.use_async_omni_output = False
        self.omni_pooler_payload_include_hidden = False
        self.have_multimodal_outputs = False
        self._batch_preprocess_in_progress = False
        self._batch_req_infos: list[dict[str, Any]] = []
        self._batch_sidecar_results: list[dict[str, Any]] = []
        self._postprocess_cursor = 0
        self._logged_stage0_backend = False
        # Per-request speech-span state owned by this model instance, NOT by the
        # runner's model_intermediate_buffer. The buffer can be reset/replaced by
        # the runner between steps (e.g. _update_intermediate_buffer merging the
        # preprocess update_dict), which would clobber the values
        # postprocess_sampled_tokens writes — breaking the force_text_abos ->
        # generate_speech handoff. Keeping state here makes it survive any buffer
        # churn. Cleared per-request on finish.
        self._speech_state: dict[str, dict[str, Any]] = {}
        # Sidecar KV and generated codec history belong to the model, not
        # the runner's CPU payload buffer. Recompute only rebuilds paged LLM
        # KV; these states remain at the last committed input position.
        self._crq_gpu_state: dict[str, dict[str, Any]] = {}
        self._speech_ids_gpu_state: dict[str, torch.Tensor] = {}
        self._crq_generators: dict[str, torch.Generator] = {}
        self._replay_state: dict[str, _ReplayState] = {}
        self._crq_bound_req_id: str | None = None
        self._logprobs_mode = vllm_config.model_config.logprobs_mode

    @staticmethod
    def _move_nested_to_device(value: Any, device: torch.device) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(device=device)
        if isinstance(value, tuple):
            return tuple(FunAudioChatForConditionalGeneration._move_nested_to_device(v, device) for v in value)
        if isinstance(value, list):
            return [FunAudioChatForConditionalGeneration._move_nested_to_device(v, device) for v in value]
        return value

    def _crq_resolve_gpu_kv(
        self,
        req_id: str | None,
        cached_audio_embeds: Any,
        cached_past_key_values: Any,
        device: torch.device,
    ) -> tuple[Any, Any]:
        # Source of truth is the per-request GPU-resident state. On first use
        # (or after finish_speech freed it) seed from the incoming buffer cache
        # (moved to GPU once); steady state reuses the resident tensors with
        # no CPU round-trip. ``req_id is None`` can't happen for real requests
        # (``_speech_state`` relies on the same key) but degrades gracefully to
        # a plain per-call device move with no cross-step caching.
        state = self._crq_gpu_state.get(req_id) if req_id is not None else None
        if state is not None:
            return state["embeds"], state["pkv"]
        return (
            self._move_nested_to_device(cached_audio_embeds, device),
            self._move_nested_to_device(cached_past_key_values, device),
        )

    def _crq_persist_gpu_kv(self, req_id: str | None, embeds: Any, pkv: Any) -> None:
        if req_id is None:
            return
        self._crq_gpu_state[req_id] = {"embeds": embeds, "pkv": pkv}

    def _resolve_gpu_speech_ids(
        self,
        req_id: str | None,
        cached_speech_ids: Any,
        device: torch.device,
    ) -> torch.Tensor:
        if req_id is not None:
            resident = self._speech_ids_gpu_state.get(req_id)
            if resident is not None:
                return resident
        speech_ids = self._as_2d_long_tensor(cached_speech_ids, device)
        if req_id is not None:
            self._speech_ids_gpu_state[req_id] = speech_ids
        return speech_ids

    def _persist_gpu_speech_ids(self, req_id: str | None, speech_ids: torch.Tensor) -> None:
        if req_id is not None:
            self._speech_ids_gpu_state[req_id] = speech_ids

    @staticmethod
    def _as_long_token_tensor(value: Any, device: torch.device) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=torch.long).reshape(-1)[-1:]
        return torch.as_tensor([value], dtype=torch.long, device=device)

    @staticmethod
    def _as_2d_long_tensor(value: Any, device: torch.device) -> torch.Tensor:
        if value is None:
            return torch.empty((1, 0), dtype=torch.long, device=device)
        if isinstance(value, torch.Tensor):
            tensor = value.to(device=device, dtype=torch.long)
        else:
            tensor = torch.as_tensor(value, dtype=torch.long, device=device)
        if tensor.ndim == 0:
            tensor = tensor.reshape(1, 1)
        elif tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _patch_audio_invert_tower_sampling_step(self) -> None:
        if getattr(self.audio_invert_tower, "_vllm_omni_crq_generator_patched", False):
            return

        def _sampling_step_with_generator(
            decoder_self: nn.Module,
            logits: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            next_token_logits = logits[:, -1, :].to(copy=True, dtype=torch.float32, device=logits.device)
            next_token_scores = decoder_self.crq_logits_processor(
                torch.cat([decoder_self.crq_speech_ids, *decoder_self.crq_generate_tokens], dim=-1),
                next_token_logits,
            )

            if decoder_self.crq_do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(
                    probs,
                    num_samples=1,
                    generator=getattr(decoder_self, "crq_generator", None),
                ).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            return next_tokens, logits

        self.audio_invert_tower.sampling_step = MethodType(_sampling_step_with_generator, self.audio_invert_tower)
        self.audio_invert_tower._vllm_omni_crq_generator_patched = True

    def _empty_audio_token_ids(self, device: torch.device) -> torch.Tensor:
        return torch.full(
            (1, int(self.config.audio_config.group_size)),
            -1,
            dtype=torch.long,
            device=device,
        )

    def _build_crq_sampling_config(self) -> tuple[LogitsProcessorList, bool]:
        """Official audio sampling is independent of text SamplingParams.

        In particular, the default greedy text path must not apply the
        audio repetition penalty to text logits. User-supplied text sampling
        parameters continue to be handled by vLLM's text sampler.
        """
        defaults = _OFFICIAL_CRQ_SAMPLING_DEFAULTS
        return LogitsProcessorList(
            [
                RepetitionPenaltyLogitsProcessor(penalty=defaults["repetition_penalty"]),
                TemperatureLogitsWarper(defaults["temperature"]),
                TopPLogitsWarper(defaults["top_p"]),
            ]
        ), True

    def _get_stage0_backend(self) -> str:
        try:
            backend_cls = self.get_language_model().model.layers[0].self_attn.attn.get_attn_backend()
            backend_name = str(backend_cls.get_name())
        except (AttributeError, IndexError, TypeError):
            backend_name = "UNKNOWN"
        if not self._logged_stage0_backend:
            logger.debug("FunAudioChat stage-0 native language backend: %s", backend_name)
            self._logged_stage0_backend = True
        return backend_name

    def _run_audio_sidecar_step(
        self,
        hidden_state: torch.Tensor,
        current_input_token_id: torch.Tensor | int,
        speech_ids: torch.Tensor,
        cached_audio_embeds: Any,
        cached_past_key_values: Any,
        logits_processor: LogitsProcessorList,
        do_sample: bool,
        current_text_seq_len: int,
        req_id: str | None = None,
    ) -> dict[str, Any]:
        device = hidden_state.device
        text_embed = (
            self.get_language_model()
            .embed_input_ids(self._as_long_token_tensor(current_input_token_id, device))
            .reshape(1, 1, -1)
        )
        speech_inputs_embeds = hidden_state.reshape(1, 1, -1) + text_embed.detach()
        attention_mask = torch.ones((1, max(current_text_seq_len, 1)), dtype=torch.long, device=device)
        position_ids = torch.tensor([[max(current_text_seq_len - 1, 0)]], dtype=torch.long, device=device)

        embeds_gpu, pkv_gpu = self._crq_resolve_gpu_kv(req_id, cached_audio_embeds, cached_past_key_values, device)
        self.audio_invert_tower.crq_audio_embeds = embeds_gpu
        self.audio_invert_tower.crq_past_key_values = pkv_gpu
        self.audio_invert_tower.crq_generator = self._crq_generators.get(req_id)
        self._crq_bound_req_id = req_id
        self.audio_invert_tower.crq_do_sample = do_sample
        self.audio_invert_tower.crq_logits_processor = logits_processor
        self.audio_invert_tower.crq_speech_ids = speech_ids
        self.audio_invert_tower.crq_generate_forward(
            inputs_embeds=speech_inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
        )

        next_audio_tokens = self.audio_invert_tower.crq_generate_tokens.reshape(1, -1).to(dtype=torch.long)
        eos_token_id = int(self.config.audio_config.eos_token_id)
        finish_speech = bool((next_audio_tokens == eos_token_id).any().item())
        if finish_speech:
            next_audio_tokens = torch.full_like(next_audio_tokens, eos_token_id)

        updated_speech_ids = torch.cat([speech_ids, next_audio_tokens], dim=-1)
        self._persist_gpu_speech_ids(req_id, updated_speech_ids)
        # Persist the post-forward KV on GPU (per request); the returned
        # tensors stay on device so the runner buffer holds GPU refs (no D2H).
        embeds_out = self.audio_invert_tower.crq_audio_embeds
        pkv_out = self.audio_invert_tower.crq_past_key_values
        self._crq_persist_gpu_kv(req_id, embeds_out, pkv_out)
        return {
            _AUDIO_TOKEN_IDS_KEY: next_audio_tokens.detach(),
            _CRQ_AUDIO_EMBEDS_KEY: embeds_out,
            _CRQ_PAST_KEY_VALUES_KEY: pkv_out,
            _FINISH_SPEECH_KEY: finish_speech,
            _SPEECH_IDS_KEY: updated_speech_ids,
        }

    def _run_audio_sidecar_decode_warmup(
        self,
        hidden_state: torch.Tensor,
        current_input_token_id: torch.Tensor | int,
        speech_ids: torch.Tensor,
        cached_audio_embeds: Any,
        cached_past_key_values: Any,
        logits_processor: LogitsProcessorList,
        do_sample: bool,
        req_id: str | None = None,
    ) -> dict[str, Any]:
        device = hidden_state.device
        text_embed = (
            self.get_language_model()
            .embed_input_ids(self._as_long_token_tensor(current_input_token_id, device))
            .reshape(1, 1, -1)
        )
        speech_inputs_embeds = hidden_state.reshape(1, 1, -1) + text_embed.detach()

        embeds_gpu, pkv_gpu = self._crq_resolve_gpu_kv(req_id, cached_audio_embeds, cached_past_key_values, device)
        self.audio_invert_tower.crq_audio_embeds = embeds_gpu
        self.audio_invert_tower.crq_past_key_values = pkv_gpu
        self.audio_invert_tower.crq_generator = self._crq_generators.get(req_id)
        self._crq_bound_req_id = req_id
        self.audio_invert_tower.crq_do_sample = do_sample
        self.audio_invert_tower.crq_logits_processor = logits_processor
        self.audio_invert_tower.crq_speech_ids = speech_ids
        self.audio_invert_tower.crq_generate_forward(
            inputs_embeds=speech_inputs_embeds,
            return_dict=True,
        )
        embeds_out = self.audio_invert_tower.crq_audio_embeds
        pkv_out = self.audio_invert_tower.crq_past_key_values
        self._crq_persist_gpu_kv(req_id, embeds_out, pkv_out)
        return {
            _CRQ_AUDIO_EMBEDS_KEY: embeds_out,
            _CRQ_PAST_KEY_VALUES_KEY: pkv_out,
        }

    def _run_audio_sidecar_prefill_warmup(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        speech_ids: torch.Tensor,
        cached_audio_embeds: Any,
        cached_past_key_values: Any,
        logits_processor: LogitsProcessorList,
        do_sample: bool,
        req_id: str | None = None,
    ) -> dict[str, Any]:
        device = hidden_states.device
        # Only the original prompt reaches this path. KV recomputation
        # replays saved LLM embeddings without touching this sidecar state.
        if req_id is not None and req_id in self._crq_gpu_state:
            raise RuntimeError(f"FunAudioChat prompt sidecar was already initialized for {req_id}")
        input_ids = input_ids.to(device=device, dtype=torch.long).reshape(1, -1)
        text_embeds = (
            self.get_language_model()
            .embed_input_ids(input_ids.reshape(-1))
            .reshape(
                1,
                -1,
                hidden_states.shape[-1],
            )
        )
        speech_inputs_embeds = hidden_states.reshape(1, -1, hidden_states.shape[-1]) + text_embeds.detach()

        embeds_gpu, pkv_gpu = self._crq_resolve_gpu_kv(req_id, cached_audio_embeds, cached_past_key_values, device)
        self.audio_invert_tower.crq_audio_embeds = embeds_gpu
        self.audio_invert_tower.crq_past_key_values = pkv_gpu
        self.audio_invert_tower.crq_generator = self._crq_generators.get(req_id)
        self._crq_bound_req_id = req_id
        self.audio_invert_tower.crq_do_sample = do_sample
        self.audio_invert_tower.crq_logits_processor = logits_processor
        self.audio_invert_tower.crq_speech_ids = speech_ids
        self.audio_invert_tower.crq_generate_forward(
            inputs_embeds=speech_inputs_embeds,
            return_dict=True,
        )
        embeds_out = self.audio_invert_tower.crq_audio_embeds
        pkv_out = self.audio_invert_tower.crq_past_key_values
        self._crq_persist_gpu_kv(req_id, embeds_out, pkv_out)
        return {
            _CRQ_AUDIO_EMBEDS_KEY: embeds_out,
            _CRQ_PAST_KEY_VALUES_KEY: pkv_out,
        }

    def _gather_user_audio_embeds(
        self,
        mm_features: Any,
        device: torch.device,
    ) -> tuple[torch.Tensor, ...] | None:
        """Produce per-item audio embeddings from the user-uploaded audio.

        ``mm_features`` is the request's ``list[MultiModalFeatureSpec]`` the
        runner forwards into the prefill span (see
        ``wants_mm_features_in_preprocess``). Each feature's processed data
        carries the keys produced by ``FunAudioChatMultiModalProcessor``:
        ``speech_ids``/``speech_attention_mask`` (discrete codec path) and
        ``input_features``/``feature_attention_mask``/``feature_exist_mask``
        (continuous Whisper-mel path). We hand all of them to the inherited
        native ``embed_multimodal``, which runs the discrete + continuous
        audio towers and returns one ``(num_features_i, output_dim)`` tensor
        per audio item, in prompt order.

        Returns ``None`` when there is no audio item in this request (a
        text-only prompt) so the caller keeps the plain text embeddings.
        """
        if not mm_features:
            return None

        # One MultiModalFeatureSpec per audio item; order them by their
        # placeholder offset so the returned tuple matches prompt order
        # (matters when limit_mm_per_prompt.audio > 1).
        audio_features = sorted(
            (f for f in mm_features if getattr(f, "modality", "") == "audio"),
            key=lambda f: getattr(getattr(f, "mm_position", None), "offset", 0),
        )
        if not audio_features:
            return None

        keys = {
            "speech_ids",
            "speech_attention_mask",
            "input_features",
            "feature_attention_mask",
            "feature_exist_mask",
        }
        # MultiModalFeatureSpec.gather_kwargs returns dict[key -> list[tensor]],
        # one per-item slice (MultiModalBatchedField splits the processor's
        # batched tensor along dim 0). The native embed_multimodal reads:
        #   - speech_ids / speech_attention_mask: accepts a list of 1D tensors
        #     (it pads them itself) -> pass the list through unchanged.
        #   - input_features / feature_attention_mask / feature_exist_mask: the
        #     native code asserts isinstance(..., torch.Tensor) and expects a
        #     batched tensor (N, ...). So we re-stack the per-item slices with
        #     torch.stack(dim=0), matching how the runner's mm branch would
        #     have batched them. Move to the compute device while stacking.
        gathered = MultiModalFeatureSpec.gather_kwargs(audio_features, keys)
        if "speech_ids" not in gathered:
            return None
        kwargs: dict[str, Any] = {
            "speech_ids": [t.to(device=device) for t in gathered["speech_ids"]],
            "speech_attention_mask": (
                [t.to(device=device) for t in gathered["speech_attention_mask"]]
                if "speech_attention_mask" in gathered
                else None
            ),
        }
        for tensor_key in ("input_features", "feature_attention_mask", "feature_exist_mask"):
            items = gathered.get(tensor_key)
            if not items:
                kwargs[tensor_key] = None
                continue
            kwargs[tensor_key] = torch.stack([t.to(device=device) for t in items], dim=0)
            if tensor_key == "feature_exist_mask":
                kwargs[tensor_key] = kwargs[tensor_key].reshape(-1)

        embeds = self.embed_multimodal(**kwargs)
        if embeds is None or len(embeds) == 0:
            return None
        return tuple(embeds)

    def _feedback_embeds(self, text_embeds: torch.Tensor, codes: torch.Tensor | None) -> torch.Tensor:
        if codes is None:
            return text_embeds
        features = self.audio_tower(codes.to(device=text_embeds.device, dtype=torch.long))
        if isinstance(features, BaseModelOutput):
            features = features.last_hidden_state
        elif isinstance(features, (tuple, list)):
            features = features[0]
        return (text_embeds + features.reshape_as(text_embeds)) / 2

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        del input_embeds
        if not self._batch_preprocess_in_progress:
            self._batch_req_infos = []
            self._batch_sidecar_results = []
            self._postprocess_cursor = 0
            self._batch_preprocess_in_progress = True

        span_len = int(input_ids.numel())
        if span_len == 0:
            raise ValueError("FunAudioChat cannot preprocess an empty request span")
        device = input_ids.device
        req_id = info_dict.get("request_id")
        replay = self._replay_state.get(req_id)
        # The engine tags the final output stage in additional_information;
        # the runner preserves this field when assembling preprocess kwargs.
        # Text-only requests end at stage 0 and do not invoke postprocess for
        # downstream payloads, so their progress must not depend on CRQ warmup.
        final_stage = info_dict.get("omni_final_stage_id")
        speech_enabled = (
            replay.speech_enabled
            if replay is not None
            else not self.sp_gen_kwargs["disable_speech"] and (final_stage is None or int(final_stage) > 0)
        )
        seed = info_dict.get("_omni_seed")
        if speech_enabled and req_id is not None and seed is not None and req_id not in self._crq_generators:
            self._crq_generators[req_id] = torch.Generator(device=device).manual_seed(int(seed))
        start = int(info_dict.get("_omni_num_computed_tokens", replay.processed_until if replay else 0))
        end = start + span_len
        prompt_len = int(info_dict.get("_omni_prompt_len", replay.prompt_len if replay else span_len))
        initial_prefill = replay is None and start == 0 and (prompt_len > 1 or "_omni_prompt_len" in info_dict)
        if replay is None and not initial_prefill and "_omni_num_computed_tokens" in info_dict:
            raise RuntimeError(f"FunAudioChat request {req_id} has no replay history for position {start}")
        if initial_prefill and (start != 0 or end != prompt_len):
            raise ValueError("FunAudioChat requires a complete initial prefill; disable chunked prefill")
        if replay is not None and (start > replay.processed_until or end > replay.processed_until + 1):
            raise RuntimeError(
                f"FunAudioChat request {req_id} scheduled [{start}, {end}) beyond "
                f"its next live position {replay.processed_until}"
            )
        has_live_step = replay is None or end > replay.processed_until
        ss = self._speech_state.get(req_id, {})
        generate_speech = speech_enabled and bool(
            ss.get(_GENERATE_SPEECH_KEY, info_dict.get(_GENERATE_SPEECH_KEY, False))
        )
        force_audio_bos_pending = speech_enabled and bool(
            ss.get(_FORCE_AUDIO_BOS_KEY, info_dict.get(_FORCE_AUDIO_BOS_KEY, self.sp_gen_kwargs["force_text_abos"]))
        )
        speech_ids = (
            self._resolve_gpu_speech_ids(req_id, info_dict.get(_SPEECH_IDS_KEY), device)
            if speech_enabled
            else torch.empty((1, 0), dtype=torch.long, device=device)
        )
        text_embeds = self.get_language_model().embed_input_ids(input_ids.reshape(-1))
        req_embeds = text_embeds.clone()

        if initial_prefill:
            mm_embeds = self._gather_user_audio_embeds(info_dict.get("mm_features"), device)
            if mm_embeds is not None:
                flat_ids = input_ids.reshape(-1)
                is_multimodal = flat_ids == int(self.config.audio_token_index)
                expected = int(is_multimodal.sum().item())
                got = int(sum(e.shape[0] for e in mm_embeds))
                if expected != got:
                    raise ValueError(
                        f"FunAudioChat prefill audio placeholder count ({expected}) "
                        f"does not match audio embedding rows ({got}) for request {req_id}."
                    )
                if expected:
                    req_embeds = self.embed_input_ids(
                        flat_ids, multimodal_embeddings=mm_embeds, is_multimodal=is_multimodal
                    )
            if req_id is not None:
                replay = _ReplayState(
                    prompt_len=prompt_len,
                    prompt_embeds=req_embeds.detach().clone(),
                    speech_enabled=speech_enabled,
                )
                self._replay_state[req_id] = replay
        else:
            # Paged KV may have been evicted, but the CRQ cache was not. Feed
            # every old position exactly the codec feedback it originally saw.
            if replay is not None:
                prompt_end = min(end, replay.prompt_len)
                if start < prompt_end:
                    req_embeds[: prompt_end - start] = replay.prompt_embeds[start:prompt_end]
                for pos in range(max(start, replay.prompt_len), min(end, replay.processed_until)):
                    if pos not in replay.feedback:
                        raise RuntimeError(f"FunAudioChat request {req_id} lacks codec feedback at position {pos}")
                    row = pos - start
                    req_embeds[row : row + 1] = self._feedback_embeds(text_embeds[row : row + 1], replay.feedback[pos])
            if has_live_step:
                group_size = int(self.config.audio_config.group_size)
                feedback = None
                if generate_speech and speech_ids.shape[-1] >= group_size:
                    feedback = speech_ids[:, -group_size:].detach().clone()
                req_embeds[-1:] = self._feedback_embeds(text_embeds[-1:], feedback)
                if replay is not None:
                    replay.feedback[end - 1] = feedback

        if replay is not None and start < replay.processed_until and not replay.replay_logged:
            logger.info(
                "FunAudioChat stage 0 replay: request_id=%s, scheduled=[%d, %d), replayed=[%d, %d), processed_until=%d",
                req_id,
                start,
                end,
                start,
                min(end, replay.processed_until),
                replay.processed_until,
            )
            replay.replay_logged = True

        self._get_stage0_backend()
        # Only a small codec delta crosses the runner buffer. Full history,
        # prompt embeddings, current token and CRQ cache stay model-local.
        update_dict = {
            "request_id": req_id,
            _TEXT_SEQ_LEN_KEY: end,
            "audio_token_ids": self._empty_audio_token_ids(torch.device("cpu")),
        }
        self._batch_req_infos.append(
            {
                "req_id": req_id,
                "has_live_step": has_live_step,
                "initial_prefill": initial_prefill,
                "speech_enabled": speech_enabled,
                "end_position": end,
                _CURRENT_INPUT_TOKEN_ID_KEY: input_ids.reshape(-1)[-1:].detach().clone(),
                _FORCE_AUDIO_BOS_KEY: force_audio_bos_pending,
                _GENERATE_SPEECH_KEY: generate_speech,
                _TEXT_INPUT_IDS_KEY: input_ids.detach().clone() if initial_prefill else None,
                _TEXT_SEQ_LEN_KEY: end,
            }
        )
        return input_ids, req_embeds, update_dict

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: Any = None,
    ) -> torch.Tensor | None:
        del sampling_metadata  # Text parameters do not configure the CRQ sampler.
        logits = super().compute_logits(hidden_states)
        if logits is None:
            self._batch_preprocess_in_progress = False
            return None

        self._batch_sidecar_results = []
        for idx, info in enumerate(self._batch_req_infos):
            req_id = info["req_id"]
            result = {
                "req_id": req_id,
                "has_live_step": info["has_live_step"],
                "speech_enabled": info["speech_enabled"],
                "end_position": info["end_position"],
                _FORCE_AUDIO_BOS_KEY: info[_FORCE_AUDIO_BOS_KEY],
                _FINISH_SPEECH_KEY: False,
                _GENERATE_SPEECH_KEY: info[_GENERATE_SPEECH_KEY],
                "audio_token_ids": self._empty_audio_token_ids(torch.device("cpu")),
            }
            if info["has_live_step"] and info["speech_enabled"]:
                processors, do_sample = self._build_crq_sampling_config()
                if info["initial_prefill"]:
                    result["_run_prefill_crq_warmup"] = True
                    result["_prefill_input_ids"] = info[_TEXT_INPUT_IDS_KEY]
                    result["_prefill_crq_logits_processor"] = processors
                    result["_prefill_crq_do_sample"] = do_sample
                else:
                    kwargs: dict[str, Any] = dict(
                        hidden_state=hidden_states[idx],
                        current_input_token_id=info[_CURRENT_INPUT_TOKEN_ID_KEY],
                        speech_ids=self._resolve_gpu_speech_ids(req_id, None, hidden_states.device),
                        cached_audio_embeds=None,
                        cached_past_key_values=None,
                        logits_processor=processors,
                        do_sample=do_sample,
                        req_id=req_id,
                    )
                    if info[_GENERATE_SPEECH_KEY]:
                        step = self._run_audio_sidecar_step(**kwargs, current_text_seq_len=info[_TEXT_SEQ_LEN_KEY])
                        result["audio_token_ids"] = step[_AUDIO_TOKEN_IDS_KEY]
                        result[_FINISH_SPEECH_KEY] = step[_FINISH_SPEECH_KEY]
                    else:
                        self._run_audio_sidecar_decode_warmup(**kwargs)
            if info["has_live_step"] and not result.get("_run_prefill_crq_warmup"):
                self._commit_input_position(req_id, info["end_position"])
            self._batch_sidecar_results.append(result)
        self._postprocess_cursor = 0
        self._batch_preprocess_in_progress = False
        return logits

    def _commit_input_position(self, req_id: str | None, end: int) -> None:
        state = self._replay_state.get(req_id) if req_id is not None else None
        if state is not None:
            state.processed_until = end

    def postprocess(self, hidden_states: torch.Tensor, **info: Any) -> dict[str, Any]:
        req_id = info.get("request_id")
        if req_id is not None:
            result = next((item for item in self._batch_sidecar_results if item["req_id"] == req_id), None)
            if result is None:
                return {}
        else:
            if self._postprocess_cursor >= len(self._batch_sidecar_results):
                return {}
            result = self._batch_sidecar_results[self._postprocess_cursor]
            self._postprocess_cursor += 1
        if result.pop("_run_prefill_crq_warmup", False):
            self._run_audio_sidecar_prefill_warmup(
                hidden_states=hidden_states,
                input_ids=result.pop("_prefill_input_ids"),
                speech_ids=self._resolve_gpu_speech_ids(result.get("req_id"), None, hidden_states.device),
                cached_audio_embeds=None,
                cached_past_key_values=None,
                logits_processor=result.pop("_prefill_crq_logits_processor"),
                do_sample=result.pop("_prefill_crq_do_sample"),
                req_id=result.get("req_id"),
            )
            self._commit_input_position(result.get("req_id"), result["end_position"])
        return {"audio_token_ids": result["audio_token_ids"]}

    def sample(self, logits: torch.Tensor, sampling_metadata: SamplingMetadata) -> SamplerOutput:
        """Force speech boundaries before sampling so token logprobs agree."""
        sampler = getattr(self, "_ar_sampler", None)
        if sampler is None:
            sampler = self._ar_sampler = FunAudioChatSampler(logprobs_mode=self._logprobs_mode)
        forced: list[int | None] = []
        for result in self._batch_sidecar_results:
            if not result.get("has_live_step", True):
                forced.append(None)
            elif result.get(_FINISH_SPEECH_KEY, False):
                forced.append(int(self.config.text_config.audio_eos_index))
            elif result.get(_FORCE_AUDIO_BOS_KEY, False):
                forced.append(int(self.config.text_config.audio_bos_index))
            else:
                forced.append(None)
        # Pure replay still follows the framework's discarded-text-sample
        # contract (including its seeded-generator offset rollback). The CRQ
        # sampler and speech state advance only for an actual live position.
        output = sampler(logits=logits, sampling_metadata=sampling_metadata, forced_token_ids=forced)
        self.postprocess_sampled_tokens(
            output.sampled_token_ids,
            [info["req_id"] for info in self._batch_req_infos],
            {info["req_id"]: idx for idx, info in enumerate(self._batch_req_infos)},
            {result["req_id"]: result for result in self._batch_sidecar_results},
        )
        return output

    def postprocess_sampled_tokens(
        self,
        sampled_token_ids: torch.Tensor,
        req_ids: list[str],
        req_id_to_index: dict[str, int],
        model_intermediate_buffer: dict[str, dict[str, Any]],
    ) -> torch.Tensor:
        """Commit speech transitions without rewriting sampled token IDs."""
        if sampled_token_ids.numel() == 0:
            return sampled_token_ids
        if sampled_token_ids.ndim == 2 and sampled_token_ids.shape[-1] != 1:
            raise ValueError("FunAudioChat does not support speculative decoding")
        audio_bos_id = int(self.config.text_config.audio_bos_index)
        for rid in req_ids:
            step = model_intermediate_buffer.get(rid) or {}
            if not step.get("has_live_step", True) or not step.get("speech_enabled", True):
                continue
            idx = req_id_to_index.get(rid)
            if idx is None:
                continue
            token = sampled_token_ids[idx] if sampled_token_ids.ndim == 1 else sampled_token_ids[idx, 0]
            token_id = int(token.item())
            ss = self._speech_state.setdefault(rid, {})
            active = bool(ss.get(_GENERATE_SPEECH_KEY, step.get(_GENERATE_SPEECH_KEY, False)))
            # The official loop ends speech on a CRQ codec EOS. A text-model
            # audio_eos token alone must not stop codec generation or feedback.
            # A codec EOS also takes precedence over a same-step audio BOS.
            finished = bool(step.get(_FINISH_SPEECH_KEY, False))
            ss[_GENERATE_SPEECH_KEY] = (active or token_id == audio_bos_id) and not finished
            ss[_FORCE_AUDIO_BOS_KEY] = False
        return sampled_token_ids

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        finished = set(finished_req_ids)
        for req_id in finished:
            self._crq_gpu_state.pop(req_id, None)
            self._speech_ids_gpu_state.pop(req_id, None)
            self._speech_state.pop(req_id, None)
            self._crq_generators.pop(req_id, None)
            self._replay_state.pop(req_id, None)
        # These are previous-step scratch buffers; the runner calls cleanup
        # before the next preprocess. Release references even while idle.
        self._batch_req_infos = [item for item in self._batch_req_infos if item.get("req_id") not in finished]
        self._batch_sidecar_results = [
            item for item in self._batch_sidecar_results if item.get("req_id") not in finished
        ]
        self._batch_preprocess_in_progress = False
        self._postprocess_cursor = 0
        if self._crq_bound_req_id in finished:
            for name in (
                "crq_audio_embeds",
                "crq_past_key_values",
                "crq_speech_ids",
                "crq_generate_tokens",
                "crq_generator",
            ):
                setattr(self.audio_invert_tower, name, None)
            self._crq_bound_req_id = None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Native audio attention stores Q/K/V in one parameter. Map these
        # names before AutoWeightsLoader strips the attention module prefix;
        # the native mapper matches '.q_proj', which no longer matches the
        # local 'q_proj' name when invoked recursively in current vLLM.
        params = dict(self.named_parameters())
        loaded_stacked: set[str] = set()

        def remaining_weights():
            for name, weight in weights:
                for shard in ("q", "k", "v"):
                    target = name.replace(f".{shard}_proj.", ".qkv_proj.")
                    if (
                        target != name
                        and target in params
                        and name.startswith(("continuous_audio_tower.", "audio_tower."))
                    ):
                        param = params[target]
                        param.weight_loader(param, weight, shard)
                        loaded_stacked.add(target)
                        break
                else:
                    yield name, weight

        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(remaining_weights())
        return loaded | loaded_stacked
