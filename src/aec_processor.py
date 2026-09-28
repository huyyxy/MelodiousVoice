#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回声消除处理器模块

提供回声消除的抽象基类和具体实现：
- AECProcessor: 抽象基类
- NKFAECProcessor: 基于 NKF 模型的回声消除处理器
- NLMSAECProcessor: 基于 NLMS 自适应滤波的回声消除处理器（类似 WebRTC AEC）
"""

import os
from abc import ABC, abstractmethod
import numpy as np
import torch

from nkf import NKF
from nkf_streaming import NKFStreaming

try:
    from padasip.filters import FilterNLMS
    HAS_PADASIP = True
except ImportError:
    HAS_PADASIP = False


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


class NLMSAECProcessor(AECProcessor):
    """
    基于 NLMS（归一化最小均方）自适应滤波的回声消除处理器
    
    使用 NLMS 算法进行回声消除，这是 WebRTC AEC 的核心算法之一。
    NLMS 通过自适应地估计回声路径的冲激响应来消除回声。
    
    特点：
    - 计算效率高，适合实时处理
    - 收敛速度快于 LMS
    - 无需 GPU，纯 CPU 处理
    - 支持流式处理
    """
    
    def __init__(
        self,
        hop_size: int = 256,
        filter_length: int = 2048,
        mu: float = 0.5,
        eps: float = 1e-6,
        sample_rate: int = 16000
    ):
        """
        初始化 NLMS 回声消除处理器
        
        Args:
            hop_size: 每次处理的样本数
            filter_length: 自适应滤波器长度（采样点数），决定了可消除的最大回声延迟
                          建议设置为约 128ms 的样本数（16kHz 采样率下约 2048）
            mu: 步长参数（学习率），范围 0-2，越大收敛越快但可能不稳定
            eps: 正则化项，防止除零
            sample_rate: 采样率（仅用于信息显示）
        """
        if not HAS_PADASIP:
            raise ImportError(
                "NLMSAECProcessor 需要 padasip 库，请安装: pip install padasip"
            )
        
        self.hop_size = hop_size
        self.filter_length = filter_length
        self.mu = mu
        self.eps = eps
        self.sample_rate = sample_rate
        
        # 初始化滤波器
        self._init_filter()
        
        # 参考信号历史缓冲区（用于构建滤波器输入向量）
        self.ref_history = np.zeros(filter_length, dtype=np.float32)
    
    def _init_filter(self):
        """初始化 NLMS 滤波器"""
        self.filter = FilterNLMS(
            n=self.filter_length,
            mu=self.mu,
            eps=self.eps,
            w="zeros"
        )
    
    def process_chunk(self, mic_chunk: np.ndarray, ref_chunk: np.ndarray) -> np.ndarray:
        """
        处理一个音频块
        
        算法流程：
        1. 更新参考信号历史
        2. 对于每个样本：
           a. 使用滤波器预测回声
           b. 从麦克风信号中减去预测的回声
           c. 使用误差信号更新滤波器权重
        
        Args:
            mic_chunk: 麦克风输入（包含回声的近端信号）
            ref_chunk: 参考信号（远端信号，播放到扬声器的音频）
            
        Returns:
            回声消除后的音频块
        """
        if len(mic_chunk) != self.hop_size:
            raise ValueError(f"mic_chunk 长度必须为 {self.hop_size}，实际为 {len(mic_chunk)}")
        if len(ref_chunk) != self.hop_size:
            raise ValueError(f"ref_chunk 长度必须为 {self.hop_size}，实际为 {len(ref_chunk)}")
        
        # 确保输入是 float32
        mic = mic_chunk.astype(np.float32)
        ref = ref_chunk.astype(np.float32)
        
        # 输出缓冲区
        output = np.zeros(self.hop_size, dtype=np.float32)
        
        # 逐样本处理
        for i in range(self.hop_size):
            # 更新参考信号历史（移位并添加新样本）
            self.ref_history = np.roll(self.ref_history, 1)
            self.ref_history[0] = ref[i]
            
            # 使用 NLMS 滤波器预测回声
            echo_estimate = self.filter.predict(self.ref_history)
            
            # 误差信号 = 麦克风信号 - 预测回声（这就是回声消除后的信号）
            error = mic[i] - echo_estimate
            output[i] = error
            
            # 更新滤波器权重
            self.filter.adapt(mic[i], self.ref_history)
        
        return output
    
    def reset_state(self):
        """重置内部状态"""
        self._init_filter()
        self.ref_history.fill(0)
    
    def get_name(self) -> str:
        return "NLMS-AEC"
    
    def get_filter_info(self) -> dict:
        """
        获取滤波器信息
        
        Returns:
            包含滤波器参数和状态的字典
        """
        max_delay_ms = self.filter_length / self.sample_rate * 1000
        return {
            "filter_length": self.filter_length,
            "max_delay_ms": max_delay_ms,
            "mu": self.mu,
            "eps": self.eps,
            "hop_size": self.hop_size
        }


def create_aec_processor(type: str, **kwargs) -> AECProcessor:
    """
    工厂函数：创建回声消除处理器
    
    Args:
        type: 处理器类型，支持 "nkf"、"nlms" 或 "passthrough"
        **kwargs: 传递给对应处理器的参数
            - NKFAECProcessor (type="nkf"):
                - model_path: str, NKF 模型权重文件路径
                - block_size: int, STFT 窗口大小 (默认 1024)
                - hop_size: int, 每次处理的样本数 (默认 256)
                - device: str, 计算设备 (默认 'cpu')
            - NLMSAECProcessor (type="nlms"):
                - hop_size: int, 每次处理的样本数 (默认 256)
                - filter_length: int, 滤波器长度 (默认 2048)
                - mu: float, 步长参数 (默认 0.5)
                - eps: float, 正则化项 (默认 1e-6)
                - sample_rate: int, 采样率 (默认 16000)
            - PassthroughProcessor (type="passthrough"):
                - hop_size: int, 每次处理的样本数 (默认 256)
        
    Returns:
        AECProcessor 实例
        
    Raises:
        ValueError: 不支持的处理器类型
        ImportError: 缺少必要的依赖库
    """
    processor_map = {
        "nkf": NKFAECProcessor,
        "nlms": NLMSAECProcessor,
        "passthrough": PassthroughProcessor,
    }
    
    if type not in processor_map:
        supported_types = ", ".join(processor_map.keys())
        raise ValueError(f"不支持的处理器类型: {type}，支持的类型: {supported_types}")
    
    processor_class = processor_map[type]
    return processor_class(**kwargs)

