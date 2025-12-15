#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回声消除处理器模块

提供回声消除的抽象基类和具体实现：
- AECProcessor: 抽象基类
- NKFAECProcessor: 基于 NKF 模型的回声消除处理器
"""

import os
from abc import ABC, abstractmethod
import numpy as np
import torch

from nkf import NKF
from nkf_streaming import NKFStreaming


class AECProcessor(ABC):
    """回声消除处理器抽象基类"""
    
    @abstractmethod
    def process_chunk(self, mic_chunk: np.ndarray, ref_chunk: np.ndarray) -> np.ndarray:
        """
        处理一个音频块
        
        Args:
            mic_chunk: 麦克风输入 (numpy array)
            ref_chunk: 参考信号 (numpy array)
            
        Returns:
            回声消除后的音频块 (numpy array)
        """
        pass
    
    @abstractmethod
    def reset_state(self):
        """重置内部状态"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """获取处理器名称"""
        pass


class NKFAECProcessor(AECProcessor):
    """
    基于 NKF（Neural Kalman Filtering）的回声消除处理器
    
    使用 NKF 模型进行流式回声消除处理。
    """
    
    def __init__(
        self,
        model_path: str,
        block_size: int = 1024,
        hop_size: int = 256,
        device: str = 'cpu'
    ):
        """
        初始化 NKF 回声消除处理器
        
        Args:
            model_path: NKF 模型权重文件路径
            block_size: STFT 窗口大小
            hop_size: 每次处理的样本数
            device: 计算设备 ('cpu' 或 'cuda')
        """
        self.block_size = block_size
        self.hop_size = hop_size
        self.device = torch.device(device)
        
        # 加载模型
        self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """加载 NKF 模型"""
        # 处理模型路径
        if not os.path.isabs(model_path) and not os.path.exists(model_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path_in_script_dir = os.path.join(script_dir, model_path)
            if os.path.exists(model_path_in_script_dir):
                model_path = model_path_in_script_dir
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件未找到: {model_path}")
        
        # 创建模型
        self.model = NKF(L=4)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=True)
        self.model.to(self.device)
        self.model.eval()
        
        # 初始化流式处理器
        self._init_streaming()
    
    def _init_streaming(self):
        """初始化流式处理器"""
        self.aec_stream = NKFStreaming(
            self.model,
            block_size=self.block_size,
            hop_size=self.hop_size
        )
        self.aec_stream.to(self.device)
    
    def process_chunk(self, mic_chunk: np.ndarray, ref_chunk: np.ndarray) -> np.ndarray:
        """
        处理一个音频块
        
        Args:
            mic_chunk: 麦克风输入 (numpy array, shape: [hop_size])
            ref_chunk: 参考信号 (numpy array, shape: [hop_size])
            
        Returns:
            回声消除后的音频块 (numpy array, shape: [hop_size])
        """
        # 确保输入长度正确
        if len(mic_chunk) != self.hop_size:
            raise ValueError(f"mic_chunk 长度必须为 {self.hop_size}，实际为 {len(mic_chunk)}")
        if len(ref_chunk) != self.hop_size:
            raise ValueError(f"ref_chunk 长度必须为 {self.hop_size}，实际为 {len(ref_chunk)}")
        
        # 转换为 tensor
        x_tensor = torch.from_numpy(ref_chunk.astype(np.float32)).to(self.device)
        y_tensor = torch.from_numpy(mic_chunk.astype(np.float32)).to(self.device)
        
        # 调用 NKF 流式处理
        with torch.no_grad():
            output_tensor = self.aec_stream.process_chunk(x_tensor, y_tensor)
        
        return output_tensor.cpu().numpy()
    
    def reset_state(self):
        """重置内部状态（重新初始化流式处理器）"""
        self._init_streaming()
    
    def get_name(self) -> str:
        return "NKF-AEC"


class PassthroughProcessor(AECProcessor):
    """
    直通处理器（用于测试）
    
    直接返回麦克风输入，不做任何处理。
    """
    
    def __init__(self, hop_size: int = 256):
        self.hop_size = hop_size
    
    def process_chunk(self, mic_chunk: np.ndarray, ref_chunk: np.ndarray) -> np.ndarray:
        """直接返回麦克风输入"""
        return mic_chunk.copy()
    
    def reset_state(self):
        """无需重置"""
        pass
    
    def get_name(self) -> str:
        return "Passthrough"


def create_aec_processor(
    model_path: str,
    block_size: int = 1024,
    hop_size: int = 256,
    device: str = 'cpu',
    passthrough: bool = False
) -> AECProcessor:
    """
    工厂函数：创建回声消除处理器
    
    Args:
        model_path: NKF 模型权重文件路径
        block_size: STFT 窗口大小
        hop_size: 每次处理的样本数
        device: 计算设备
        passthrough: 是否使用直通模式（用于测试）
        
    Returns:
        AECProcessor 实例
    """
    if passthrough:
        return PassthroughProcessor(hop_size=hop_size)
    else:
        return NKFAECProcessor(
            model_path=model_path,
            block_size=block_size,
            hop_size=hop_size,
            device=device
        )

