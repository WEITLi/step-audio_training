"""
Configuration module for Step-Audio
"""

from .prompts import TTS_SYSTEM_PROMPTS, AUDIO_EDIT_SYSTEM_PROMPT
from .edit_config import get_supported_edit_types

__all__ = [
    'TTS_SYSTEM_PROMPTS',
    'AUDIO_EDIT_SYSTEM_PROMPT',
    'get_supported_edit_types'
]