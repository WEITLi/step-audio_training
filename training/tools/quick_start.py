#!/usr/bin/env python3
"""
Quick start script for training
Provides an interactive setup wizard
"""

import os
import sys
from pathlib import Path
import subprocess


def print_header(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def check_paths():
    """Check if required paths exist"""
    print_header("Checking Paths")
    
    paths = {
        'CosyVoice': '/Users/weitao_li/CodeField/DCAI/Projects/CosyVoice',
        'Step-Audio-EditX': '/Users/weitao_li/CodeField/DCAI/Projects/Step-Audio-EditX',
        'Tokenizer': '/Users/weitao_li/CodeField/DCAI/Projects/Step-Audio-EditX/pretrained_models/Step-Audio-Tokenizer',
    }
    
    all_exist = True
    for name, path in paths.items():
        if Path(path).exists():
            print(f"✓ {name}: {path}")
        else:
            print(f"✗ {name} not found: {path}")
            all_exist = False
    
    return all_exist


def check_data():
    """Check if data is prepared"""
    print_header("Checking Data")
    
    data_dir = Path('/Users/weitao_li/CodeField/DCAI/Projects/Step-Audio-EditX/data')
    
    required_files = {
        'train/wav.scp': 'Training audio list',
        'train/text': 'Training transcriptions',
        'dev/wav.scp': 'Validation audio list',
        'dev/text': 'Validation transcriptions',
    }
    
    all_exist = True
    for file, desc in required_files.items():
        file_path = data_dir / file
        if file_path.exists():
            # Count lines
            with open(file_path) as f:
                count = sum(1 for _ in f)
            print(f"✓ {desc}: {count} entries")
        else:
            print(f"✗ {desc} not found: {file_path}")
            all_exist = False
    
    return all_exist


def suggest_next_steps(paths_ok, data_ok):
    """Suggest next steps based on checks"""
    print_header("Next Steps")
    
    if not paths_ok:
        print("1. Download CosyVoice repository:")
        print("   cd /Users/weitao_li/CodeField/DCAI/Projects")
        print("   git clone https://github.com/FunAudioLLM/CosyVoice.git")
        print()
        print("2. Download Step-Audio-Tokenizer:")
        print("   cd /Users/weitao_li/CodeField/DCAI/Projects/Step-Audio-EditX")
        print("   # Download from HuggingFace or ModelScope")
        return
    
    if not data_ok:
        print("Prepare your training data:")
        print()
        print("1. Create data directories:")
        print("   mkdir -p data/train data/dev")
        print()
        print("2. Prepare wav.scp (format: utt_id /path/to/audio.wav):")
        print("   utt001 /path/to/audio/utt001.wav")
        print("   utt002 /path/to/audio/utt002.wav")
        print()
        print("3. Prepare text (format: utt_id transcription):")
        print("   utt001 这是第一句话")
        print("   utt002 这是第二句话")
        print()
        print("4. Then run:")
        print("   bash training/scripts/train_flow.sh --stage 0 --stop_stage 3")
        return
    
    print("✓ All checks passed! Ready to start training.")
    print()
    print("Run the training pipeline:")
    print("  cd /Users/weitao_li/CodeField/DCAI/Projects/Step-Audio-EditX")
    print()
    print("Option 1: Full pipeline (data preparation + training)")
    print("  bash training/scripts/train_flow.sh")
    print()
    print("Option 2: Step by step")
    print("  # Data preparation")
    print("  bash training/scripts/train_flow.sh --stage 0 --stop_stage 3")
    print()
    print("  # Training")
    print("  bash training/scripts/train_flow.sh --stage 4 --stop_stage 4")
    print()
    print("  # Export model")
    print("  bash training/scripts/train_flow.sh --stage 5 --stop_stage 6")


def main():
    print("\n" + "🚀" * 30)
    print("  Step-Audio-EditX Flow Model Training")
    print("  Quick Start Wizard")
    print("🚀" * 30)
    
    paths_ok = check_paths()
    data_ok = check_data()
    
    suggest_next_steps(paths_ok, data_ok)
    
    print("\n" + "=" * 60)
    print("For more information, see:")
    print("  training/README.md")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
