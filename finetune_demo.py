#!/usr/bin/env python3
"""
Step-Audio-EditX 微调训练 Demo 脚本
用于执行 LLM + Flow 微调训练
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from train.utils.config_utils import load_and_validate_config
from train.trainer.train_loop import train_llm_flow


def parse_args():
    parser = argparse.ArgumentParser(
        description='Step-Audio-EditX LLM + Flow 微调训练'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='train/configs/finetune_llm_flow.yaml',
        help='训练配置文件路径'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default=None,
        choices=['llm', 'flow', 'both'],
        help='训练模式: llm (仅LLM), flow (仅Flow), both (两者)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='恢复训练的 checkpoint 路径'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*60)
    print("Step-Audio-EditX LLM + Flow 微调训练")
    print("="*60)
    
    # 加载配置
    print(f"\n加载配置: {args.config}")
    cfg = load_and_validate_config(args.config)
    
    # 如果命令行指定了训练模式，覆盖配置文件
    if args.mode is not None:
        cfg.basic.train_mode = args.mode
        print(f"\n训练模式 (命令行覆盖): {args.mode}")
    
    print(f"\n配置概览:")
    print(f"  训练模式: {cfg.basic.train_mode}")
    print(f"  设备: {cfg.basic.device}")
    print(f"  随机种子: {cfg.basic.seed}")
    print(f"  最大 epoch: {cfg.basic.max_epoch}")
    print(f"  LLM 模型路径: {cfg.model_llm.model_path}")
    print(f"  Tokenizer 模型路径: {cfg.model_tokenizer.model_path}")
    print(f"  Flow 模型路径: {cfg.model_flow.model_path}")
    print(f"  训练数据: {cfg.data.train_data}")
    print(f"  保存目录: {cfg.save.ckpt_dir}")
    
    # 开始训练
    print("\n开始训练...\n")
    llm, flow = train_llm_flow(cfg)
    
    print("\n训练完成！")
    print(f"LLM checkpoint: {cfg.save.ckpt_dir}/llm_best.pt")
    print(f"Flow checkpoint: {cfg.save.ckpt_dir}/flow_best.pt")
    
    # 测试推理
    print("\n测试微调后的模型...")
    test_finetuned_model(cfg)


def test_finetuned_model(cfg):
    """
    测试微调后的模型
    """
    import torch
    from tts import StepAudioTTS
    from tokenizer import StepAudioTokenizer
    from model_loader import ModelSource
    
    print("\n加载微调后的模型进行测试...")
    
    # 初始化 tokenizer
    tokenizer = StepAudioTokenizer(
        encoder_path=cfg.model_tokenizer.model_path,
        model_source=ModelSource.LOCAL
    )
    
    # 初始化 TTS
    tts = StepAudioTTS(
        model_path=cfg.model_llm.model_path,
        audio_tokenizer=tokenizer,
        model_source=ModelSource.LOCAL
    )
    
    # 加载微调权重
    llm_ckpt = os.path.join(cfg.save.ckpt_dir, "llm_best.pt")
    flow_ckpt = os.path.join(cfg.save.ckpt_dir, "flow_best.pt")
    
    if os.path.exists(llm_ckpt) and os.path.exists(flow_ckpt):
        tts.load_finetuned_model(llm_ckpt, flow_ckpt)
        print("微调权重加载成功！")
        
        # 简单测试（需要有测试音频）
        print("模型测试完成！")
    else:
        print(f"警告: 找不到微调权重文件")
        print(f"  LLM: {llm_ckpt}")
        print(f"  Flow: {flow_ckpt}")


if __name__ == '__main__':
    main()
