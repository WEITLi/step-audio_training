# Copyright (c) 2024 Stepfun AI
# Adapted from CosyVoice
#
# 数据处理器 - 实现 CosyVoice 风格的数据处理 pipeline

import logging
import random
import pyarrow.parquet as pq
from io import BytesIO
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AUDIO_FORMAT_SETS = {'flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'}


def parquet_opener(data, mode='train'):
    """读取 parquet 文件并生成样本
    
    Args:
        data: Iterable[{src: parquet_path, ...}]
        mode: 'train' or 'eval'
        
    Yields:
        样本字典
    """
    for sample in data:
        assert 'src' in sample
        url = sample['src']
        try:
            for df in pq.ParquetFile(url).iter_batches(batch_size=64):
                df = df.to_pandas()
                for i in range(len(df)):
                    sample.update(dict(df.loc[i]))
                    # 必须创建新的 dict，不能直接返回 sample
                    yield {**sample}
        except Exception as ex:
            logger.warning(f'Failed to open {url}: {ex}')


def filter(data,
           max_length=40960,
           min_length=100,
           token_max_length=200,
           token_min_length=1,
           mode='train'):
    """过滤样本长度
    
    Args:
        data: Iterable[样本]
        max_length: 最大音频长度（样本点数）
        min_length: 最小音频长度
        token_max_length: 最大 token 长度
        token_min_length: 最小 token 长度
        
    Yields:
        过滤后的样本
    """
    for sample in data:
        try:
            # 加载音频
            sample['speech'], sample['sample_rate'] = torchaudio.load(
                BytesIO(sample['audio_data'])
            )
            sample['speech'] = sample['speech'].mean(dim=0, keepdim=True)
            del sample['audio_data']
            
            # 检查音频长度
            num_samples = sample['speech'].size(1)
            if num_samples < min_length or num_samples > max_length:
                continue
            
            # 检查 token 长度
            if len(sample['speech_token']) < token_min_length:
                continue
            if len(sample['speech_token']) > token_max_length:
                continue
            
            yield sample
            
        except Exception as e:
            logger.warning(f"Filter failed for {sample.get('utt', 'unknown')}: {e}")
            continue


def resample(data, resample_rate=24000, min_sample_rate=16000, mode='train'):
    """重采样音频
    
    Args:
        data: Iterable[样本]
        resample_rate: 目标采样率
        min_sample_rate: 最小采样率阈值
        
    Yields:
        重采样后的样本
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'speech' in sample
        
        sample_rate = sample['sample_rate']
        waveform = sample['speech']
        
        if sample_rate != resample_rate:
            if sample_rate < min_sample_rate:
                continue
            sample['sample_rate'] = resample_rate
            sample['speech'] = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=resample_rate
            )(waveform)
        
        # 归一化
        max_val = sample['speech'].abs().max()
        if max_val > 1:
            sample['speech'] /= max_val
        
        yield sample


def compute_fbank(data, feat_extractor, token_mel_ratio=2, mode='train'):
    """计算 mel spectrogram
    
    Args:
        data: Iterable[样本]
        feat_extractor: mel spectrogram 提取器
        token_mel_ratio: token 和 mel 的比例（CosyVoice 使用 2）
        
    Yields:
        添加了 speech_feat 的样本
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'speech' in sample
        assert 'utt' in sample
        
        waveform = sample['speech']
        
        # 提取 mel spectrogram
        feat = feat_extractor(waveform).squeeze(dim=0).transpose(0, 1)
        
        if token_mel_ratio != 0:
            # 对齐 speech_token 和 speech_feat
            # CosyVoice: 1 token = 2 mel frames
            token_len = int(min(
                feat.shape[0] / token_mel_ratio,
                len(sample["speech_token"])
            ))
            feat = feat[:token_mel_ratio * token_len]
            sample["speech_token"] = sample["speech_token"][:token_len]
        
        sample['speech_feat'] = feat
        yield sample


def parse_embedding(data, normalize=True, mode='train'):
    """解析 embedding
    
    Args:
        data: Iterable[样本]
        normalize: 是否归一化 embedding
        
    Yields:
        添加了 tensor 格式 embedding 的样本
    """
    for sample in data:
        sample['utt_embedding'] = torch.tensor(
            sample['utt_embedding'],
            dtype=torch.float32
        )
        sample['spk_embedding'] = torch.tensor(
            sample['spk_embedding'],
            dtype=torch.float32
        )
        
        if normalize:
            sample['utt_embedding'] = F.normalize(sample['utt_embedding'], dim=0)
            sample['spk_embedding'] = F.normalize(sample['spk_embedding'], dim=0)
        
        yield sample


def tokenize(data, tokenizer, mode='train'):
    """文本 tokenization
    
    Args:
        data: Iterable[样本]
        tokenizer: Qwen tokenizer (Transformers tokenizer)
        
    Yields:
        添加了 text_token 的样本
    """
    for sample in data:
        assert 'text' in sample
        # 使用 Transformers tokenizer 进行 tokenization
        # 不使用 allowed_special 参数，因为 Transformers tokenizer 不支持
        sample['text_token'] = tokenizer.encode(
            sample['text'],
            add_special_tokens=False  # 使用 Transformers 的标准参数
        )
        yield sample


def shuffle(data, shuffle_size=1000, mode='train'):
    """局部打乱数据
    
    Args:
        data: Iterable[样本]
        shuffle_size: shuffle buffer 大小
        
    Yields:
        打乱后的样本
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= shuffle_size:
            random.shuffle(buf)
            for x in buf:
                yield x
            buf = []
    
    # 处理剩余数据
    random.shuffle(buf)
    for x in buf:
        yield x


def sort(data, sort_size=500, mode='train'):
    """按长度排序
    
    Args:
        data: Iterable[样本]
        sort_size: sort buffer 大小
        
    Yields:
        排序后的样本
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= sort_size:
            buf.sort(key=lambda x: x['speech_feat'].size(0))
            for x in buf:
                yield x
            buf = []
    
    # 处理剩余数据
    buf.sort(key=lambda x: x['speech_feat'].size(0))
    for x in buf:
        yield x


def dynamic_batch(data, max_frames_in_batch=2000, mode='train'):
    """动态 batching
    
    Args:
        data: Iterable[样本]
        max_frames_in_batch: batch 中最大帧数
        
    Yields:
        List[样本] (batch)
    """
    buf = []
    longest_frames = 0
    
    for sample in data:
        assert 'speech_feat' in sample
        assert isinstance(sample['speech_feat'], torch.Tensor)
        
        new_sample_frames = sample['speech_feat'].size(0)
        longest_frames = max(longest_frames, new_sample_frames)
        frames_after_padding = longest_frames * (len(buf) + 1)
        
        if frames_after_padding > max_frames_in_batch:
            yield buf
            buf = [sample]
            longest_frames = new_sample_frames
        else:
            buf.append(sample)
    
    if len(buf) > 0:
        yield buf


def padding(data, use_spk_embedding=False, mode='train'):
    """Padding 成 batch
    
    Args:
        data: Iterable[List[样本]]
        use_spk_embedding: 使用 spk_embedding (True) 还是 utt_embedding (False)
        
    Yields:
        batch 字典
    """
    for sample in data:
        assert isinstance(sample, list)
        
        # 按长度排序
        speech_feat_len = torch.tensor(
            [x['speech_feat'].size(0) for x in sample],
            dtype=torch.int32
        )
        order = torch.argsort(speech_feat_len, descending=True)
        
        # 提取并 padding 各个字段
        utts = [sample[i]['utt'] for i in order]
        
        # Speech token
        speech_token = [torch.tensor(sample[i]['speech_token']) for i in order]
        speech_token_len = torch.tensor([i.size(0) for i in speech_token], dtype=torch.int32)
        speech_token = pad_sequence(speech_token, batch_first=True, padding_value=0)
        
        # Speech feature (mel)
        speech_feat = [sample[i]['speech_feat'] for i in order]
        speech_feat_len = torch.tensor([i.size(0) for i in speech_feat], dtype=torch.int32)
        speech_feat = pad_sequence(speech_feat, batch_first=True, padding_value=0)
        
        # Text
        text = [sample[i]['text'] for i in order]
        text_token = [torch.tensor(sample[i]['text_token']) for i in order]
        text_token_len = torch.tensor([i.size(0) for i in text_token], dtype=torch.int32)
        text_token = pad_sequence(text_token, batch_first=True, padding_value=0)
        
        # Embedding
        utt_embedding = torch.stack([sample[i]['utt_embedding'] for i in order], dim=0)
        spk_embedding = torch.stack([sample[i]['spk_embedding'] for i in order], dim=0)
        
        batch = {
            "utts": utts,
            "speech_token": speech_token,
            "speech_token_len": speech_token_len,
            "speech_feat": speech_feat,
            "speech_feat_len": speech_feat_len,
            "text": text,
            "text_token": text_token,
            "text_token_len": text_token_len,
            "utt_embedding": utt_embedding,
            "spk_embedding": spk_embedding,
        }
        
        # 选择使用哪个 embedding
        if use_spk_embedding:
            batch["embedding"] = batch["spk_embedding"]
        else:
            batch["embedding"] = batch["utt_embedding"]
        
        yield batch
