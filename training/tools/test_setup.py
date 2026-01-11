#!/usr/bin/env python3
"""
Quick test script to verify Flow Model training setup
"""

import sys
from pathlib import Path
import torch

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, '/Users/weitao_li/CodeField/DCAI/Projects/CosyVoice')

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from cosyvoice.flow.flow import CausalMaskedDiffWithXvec
        print("✓ CosyVoice flow model")
    except Exception as e:
        print(f"✗ CosyVoice flow model: {e}")
        return False
    
    try:
        from cosyvoice.transformer.encoder import ConformerEncoder
        print("✓ Conformer encoder")
    except Exception as e:
        print(f"✗ Conformer encoder: {e}")
        return False
    
    try:
        from tokenizer import StepAudioTokenizer
        print("✓ Step-Audio tokenizer")
    except Exception as e:
        print(f"✗ Step-Audio tokenizer: {e}")
        return False
    
    return True


def test_config():
    """Test if config file can be loaded"""
    print("\nTesting config file...")
    
    try:
        from hyperpyyaml import load_hyperpyyaml
        config_path = Path(__file__).parent.parent / 'configs' / 'flow_model.yaml'
        
        if not config_path.exists():
            print(f"✗ Config file not found: {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            # Try to load without instantiating models
            content = f.read()
            print(f"✓ Config file readable ({len(content)} bytes)")
        
        return True
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False


def test_data_scripts():
    """Test if data processing scripts exist"""
    print("\nTesting data processing scripts...")
    
    scripts_dir = Path(__file__).parent.parent / 'data_processing'
    required_scripts = [
        'extract_step_audio_tokens.py',
        'make_parquet.py'
    ]
    
    all_exist = True
    for script in required_scripts:
        script_path = scripts_dir / script
        if script_path.exists():
            print(f"✓ {script}")
        else:
            print(f"✗ {script} not found")
            all_exist = False
    
    return all_exist


def test_gpu():
    """Test GPU availability"""
    print("\nTesting GPU...")
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"✓ {num_gpus} GPU(s) available")
        for i in range(num_gpus):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {name} ({mem:.1f} GB)")
        return True
    else:
        print("✗ No GPU available")
        return False


def main():
    print("=" * 60)
    print("Flow Model Training Setup Test")
    print("=" * 60)
    
    results = {
        'Imports': test_imports(),
        'Config': test_config(),
        'Data Scripts': test_data_scripts(),
        'GPU': test_gpu(),
    }
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20s}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✓ All tests passed! Ready to start training.")
        return 0
    else:
        print("\n✗ Some tests failed. Please fix the issues above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
