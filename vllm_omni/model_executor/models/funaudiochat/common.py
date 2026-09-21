# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from vllm.model_executor.models.funaudiochat import (
    FunAudioChatDummyInputsBuilder,
    FunAudioChatMultiModalProcessor,
    FunAudioChatProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY


def ensure_funaudiochat_importable() -> ModuleType:
    try:
        import funaudiochat  # type: ignore

        return funaudiochat
    except ImportError:
        pass

    env_home = os.environ.get("FUN_AUDIO_CHAT_HOME")
    extra_candidates = [Path(env_home).expanduser()] if env_home else []

    for candidate in extra_candidates:
        if candidate and candidate.exists():
            sys.path.insert(0, str(candidate))
            try:
                import funaudiochat  # type: ignore

                return funaudiochat
            except ImportError:
                continue

    raise ImportError(
        "funaudiochat package is required. Install Fun-Audio-Chat into the active "
        "environment or set FUN_AUDIO_CHAT_HOME to the repo checkout."
    )


def resolve_funaudiochat_root() -> Path:
    pkg = ensure_funaudiochat_importable()
    pkg_path = Path(pkg.__file__).resolve()
    root = pkg_path.parent.parent
    if not root.exists():
        raise FileNotFoundError(f"Resolved Fun-Audio-Chat root does not exist: {root}")
    return root


def register_funaudiochat_processor(model_cls: type[Any]) -> type[Any]:
    return MULTIMODAL_REGISTRY.register_processor(
        FunAudioChatMultiModalProcessor,
        info=FunAudioChatProcessingInfo,
        dummy_inputs=FunAudioChatDummyInputsBuilder,
    )(model_cls)
