"""
Step-Audio Tokenizer Adapter for CosyVoice Training Pipeline
Provides compatibility layer between Step-Audio-EditX tokenizer and CosyVoice data processing
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tokenizer import StepAudioTokenizer
from model_loader import ModelSource


class StepAudioTokenizerWrapper:
    """Wrapper for Step-Audio tokenizer to match CosyVoice interface"""
    
    def __init__(self, tokenizer_path):
        self.tokenizer = StepAudioTokenizer(
            encoder_path=tokenizer_path,
            model_source=ModelSource.LOCAL
        )
        # Vocabulary info
        self.vq02_vocab_size = 1024
        self.vq06_vocab_size = 1024
        self.total_vocab_size = 2048
    
    def encode(self, text):
        """
        Encode text to tokens
        Note: Step-Audio uses audio tokens, not text tokens
        This is a placeholder for compatibility
        """
        # For Step-Audio, we don't encode text to tokens
        # Text is only used for reference
        return []
    
    def decode(self, tokens):
        """Decode tokens to text (placeholder)"""
        return ""
    
    def wav2token(self, audio, sample_rate):
        """Extract tokens from audio"""
        return self.tokenizer.wav2token(audio, sample_rate)


def get_tokenizer(tokenizer_path='pretrained_models/Step-Audio-Tokenizer'):
    """
    Factory function to create tokenizer instance
    This is called by the YAML config
    """
    return StepAudioTokenizerWrapper(tokenizer_path)


if __name__ == '__main__':
    # Test the tokenizer
    import torch
    import torchaudio
    
    tokenizer = get_tokenizer()
    print(f"Tokenizer initialized")
    print(f"  VQ02 vocab size: {tokenizer.vq02_vocab_size}")
    print(f"  VQ06 vocab size: {tokenizer.vq06_vocab_size}")
    print(f"  Total vocab size: {tokenizer.total_vocab_size}")
