#!/usr/bin/env python3
"""
修复 Flow 模型的梯度问题
"""

import sys
sys.path.append('.')

import torch

def patch_flow_forward():
    """修补 Flow 模型的 forward 方法以保持梯度"""
    
    from stepvocoder.cosyvoice2.flow.flow import CausalMaskedDiffWithXvec
    from stepvocoder.cosyvoice2.utils.mask import make_pad_mask
    import torch.nn.functional as F
    
    # 保存原始的 forward 方法
    original_forward = CausalMaskedDiffWithXvec.forward
    
    def patched_forward(
        self,
        token: torch.Tensor,
        token_len: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_token_len: torch.Tensor,
        prompt_feat: torch.Tensor,
        prompt_feat_len: torch.Tensor,
        embedding: torch.Tensor,
        n_timesteps: int = 10,
    ) -> torch.Tensor:
        """修补后的 forward 方法，使用 compute_loss 进行训练"""
        assert token.shape[0] == 1
        
        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)
        
        # concat text and prompt_text
        token_len = prompt_token_len + token_len
        token = torch.concat([prompt_token, token], dim=1)
        
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask
        
        # token encode
        h, _ = self.encoder.forward(token, token_len)
        h = self.encoder_proj(h)
        
        # condition
        mel_len1 = prompt_feat.shape[1]
        mel_len2 = h.shape[1] - prompt_feat.shape[1]
        
        conds = torch.zeros_like(h)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2).contiguous()
        
        # 创建目标特征 (非 prompt 部分)
        target_feat = prompt_feat[:, mel_len1:]  # 跳过 prompt 部分
        
        # 修复：使用正确的 mel 长度创建 mask
        total_mel_len = torch.tensor([mel_len2], dtype=torch.int32, device=h.device)
        mask = (~make_pad_mask(total_mel_len)).to(h)
        
        # 使用 compute_loss 进行训练（这是正确的训练方式）
        loss, _ = self.decoder.compute_loss(
            target_feat.transpose(1, 2).contiguous(),  # (batch, time, freq) -> (batch, freq, time)
            mask.unsqueeze(1),
            h[:, mel_len1:].transpose(1, 2).contiguous(),  # 只使用非 prompt 部分
            embedding,
            cond=conds[:, :, mel_len1:],  # 只使用非 prompt 部分的条件
            streaming=False
        )
        
        return loss
    
    # 应用修补
    CausalMaskedDiffWithXvec.forward = patched_forward
    print("✅ Flow 模型 forward 方法已修补，应该能保持梯度了")

def test_patched_flow():
    """测试修补后的 Flow 模型"""
    try:
        # 先应用修补
        patch_flow_forward()
        
        from train.trainer.model_adapter import prepare_models_for_training
        from train.utils.config_utils import load_and_validate_config
        
        print("=== 测试修补后的 Flow 模型 ===")
        cfg = load_and_validate_config("train/configs/finetune_llm_flow.yaml")
        
        # 初始化模型
        llm, flow, cosy_model, text_tokenizer = prepare_models_for_training(cfg)
        
        # 测试梯度
        flow.train()
        
        # 创建测试输入
        batch_size = 1
        token_len = 12
        feat_len = 24
        
        test_token = torch.randint(0, 1000, (batch_size, token_len))
        test_token_len = torch.tensor([token_len], dtype=torch.int32)
        test_feat = torch.randn(batch_size, feat_len, 80, requires_grad=True)
        test_feat_len = torch.tensor([feat_len], dtype=torch.int32)
        test_embedding = torch.randn(batch_size, 192, requires_grad=True)
        
        print(f"输入梯度状态:")
        print(f"  test_feat requires_grad: {test_feat.requires_grad}")
        print(f"  test_embedding requires_grad: {test_embedding.requires_grad}")
        
        # 前向传播 - 现在返回损失
        loss = flow.forward(
            token=test_token,
            token_len=test_token_len,
            prompt_token=test_token[:, :5],
            prompt_token_len=torch.tensor([5], dtype=torch.int32),
            prompt_feat=test_feat[:, :10],
            prompt_feat_len=torch.tensor([10], dtype=torch.int32),
            embedding=test_embedding,
            n_timesteps=10
        )
        
        print(f"✓ 前向传播成功，损失值: {loss.item():.4f}")
        print(f"损失 requires_grad: {loss.requires_grad}")
        print(f"损失 grad_fn: {loss.grad_fn}")
        
        if loss.requires_grad:
            # 测试反向传播
            print(f"测试损失: {loss.item():.4f}")
            
            loss.backward()
            print("✅ 反向传播成功！梯度问题已修复")
            
            # 检查梯度
            grad_count = 0
            for param in flow.parameters():
                if param.grad is not None:
                    grad_count += 1
            
            print(f"有梯度的参数数量: {grad_count}")
            
        else:
            print("❌ 损失仍然不需要梯度")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_patched_flow()