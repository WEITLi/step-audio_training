# Copyright (c) 2024 Stepfun AI
# Adapted from CosyVoice
#
# Dataset 实现 - CosyVoice 风格的 IterableDataset

import random
import math
from functools import partial

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset


def read_lists(list_file):
    """读取文件列表
    
    Args:
        list_file: 包含文件路径的列表文件
        
    Returns:
        文件路径列表
    """
    lists = []
    with open(list_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if line:
                lists.append(line)
    return lists


class Processor(IterableDataset):
    """数据处理器包装类
    
    将处理函数应用到数据流上
    """
    
    def __init__(self, source, f, *args, **kw):
        assert callable(f)
        self.source = source
        self.f = f
        self.args = args
        self.kw = kw
    
    def set_epoch(self, epoch):
        self.source.set_epoch(epoch)
    
    def __iter__(self):
        """返回处理后的迭代器"""
        assert self.source is not None
        assert callable(self.f)
        return self.f(iter(self.source), *self.args, **self.kw)
    
    def apply(self, f):
        """应用新的处理函数"""
        assert callable(f)
        return Processor(self, f, *self.args, **self.kw)


class DistributedSampler:
    """分布式采样器
    
    在多 GPU/多机环境下划分数据
    """
    
    def __init__(self, shuffle=True, partition=True):
        self.epoch = -1
        self.update()
        self.shuffle = shuffle
        self.partition = partition
    
    def update(self):
        """更新分布式信息"""
        assert dist.is_available()
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.worker_id = 0
            self.num_workers = 1
        else:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
        
        return dict(
            rank=self.rank,
            world_size=self.world_size,
            worker_id=self.worker_id,
            num_workers=self.num_workers
        )
    
    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def sample(self, data):
        """采样数据
        
        Args:
            data: 数据列表
            
        Returns:
            采样后的数据列表
        """
        data = list(range(len(data)))
        
        # 分布式 partition
        if self.partition:
            if self.shuffle:
                random.Random(self.epoch).shuffle(data)
            if len(data) < self.world_size:
                data = data * math.ceil(self.world_size / len(data))
                data = data[:self.world_size]
            data = data[self.rank::self.world_size]
        
        # worker partition
        if len(data) < self.num_workers:
            data = data * math.ceil(self.num_workers / len(data))
            data = data[:self.num_workers]
        data = data[self.worker_id::self.num_workers]
        
        return data


class DataList(IterableDataset):
    """数据列表封装
    
    从文件列表中生成样本
    """
    
    def __init__(self, lists, shuffle=True, partition=True):
        self.lists = lists
        self.sampler = DistributedSampler(shuffle, partition)
    
    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)
    
    def __iter__(self):
        sampler_info = self.sampler.update()
        indexes = self.sampler.sample(self.lists)
        for index in indexes:
            data = dict(src=self.lists[index])
            data.update(sampler_info)
            yield data


def Dataset(data_list_file,
            data_pipeline,
            mode='train',
            shuffle=True,
            partition=True):
    """构建 Dataset
    
    Args:
        data_list_file: 数据列表文件路径
        data_pipeline: 数据处理 pipeline (函数列表)
        mode: 'train' or 'eval'
        shuffle: 是否打乱数据
        partition: 是否分布式划分
        
    Returns:
        IterableDataset
    """
    lists = read_lists(data_list_file)
    dataset = DataList(lists, shuffle=shuffle, partition=partition)
    
    # 应用 data pipeline
    for func in data_pipeline:
        dataset = Processor(dataset, func, mode=mode)
    
    return dataset
