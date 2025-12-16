#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
延迟估计器模块

提供延迟估计的抽象基类和具体实现：
- DelayEstimator: 抽象基类
- GCCPHATDelayEstimator: 基于 GCC-PHAT 算法的延迟估计器
- FixedDelayEstimator: 固定延迟值估计器
"""

from abc import ABC, abstractmethod
import numpy as np
from utils import gcc_phat


class DelayEstimator(ABC):
    """延迟估计器抽象基类"""
    
    @abstractmethod
    def estimate_delay(self, mic_signal: np.ndarray, ref_signal: np.ndarray, sample_rate: int) -> int:
        """
        估计延迟（采样点数）
        
        Args:
            mic_signal: 麦克风信号 (numpy array)
            ref_signal: 参考信号 (numpy array)
            sample_rate: 采样率 (Hz)
            
        Returns:
            延迟采样点数（正值表示麦克风信号落后于参考信号）
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """获取估计器名称"""
        pass


class GCCPHATDelayEstimator(DelayEstimator):
    """
    基于 GCC-PHAT 算法的延迟估计器
    
    使用广义互相关相位变换（Generalized Cross Correlation with Phase Transform）
    来估计两个信号之间的时间延迟。
    """
    
    def __init__(self, interp: int = 1, offset_seconds: float = 0.001):
        """
        初始化 GCC-PHAT 延迟估计器
        
        Args:
            interp: 插值倍数，用于提高精度
            offset_seconds: 偏移量（秒），用于补偿算法误差
        """
        self.interp = interp
        self.offset_seconds = offset_seconds
    
    def estimate_delay(self, mic_signal: np.ndarray, ref_signal: np.ndarray, sample_rate: int) -> int:
        """
        使用 GCC-PHAT 算法估计延迟
        
        Args:
            mic_signal: 麦克风信号 (numpy array)
            ref_signal: 参考信号 (numpy array)
            sample_rate: 采样率 (Hz)
            
        Returns:
            延迟采样点数
        """
        # 检查信号是否有效
        if np.abs(mic_signal).mean() < 1e-6:
            print("警告: 麦克风信号几乎为静音，无法计算延迟。")
            return 0
        
        if np.abs(ref_signal).mean() < 1e-6:
            print("警告: 参考信号几乎为静音，无法计算延迟。")
            return 0
        
        try:
            # 使用 GCC-PHAT 计算延迟
            # tau 是 mic 相对于 ref 的延迟（秒），正值表示 mic 落后于 ref
            tau = gcc_phat(mic_signal, ref_signal, fs=sample_rate, interp=self.interp)
            
            # 转换为采样点数，并应用偏移补偿
            delay_samples = max(0, int((tau - self.offset_seconds) * sample_rate))
            
            return delay_samples
            
        except Exception as e:
            print(f"警告: 延迟计算失败: {e}，使用默认延迟0")
            return 0
    
    def get_name(self) -> str:
        return "GCC-PHAT"


class FixedDelayEstimator(DelayEstimator):
    """
    固定延迟估计器
    
    返回预设的固定延迟值，适用于已知系统延迟的场景。
    """
    
    def __init__(self, delay_samples: int):
        """
        初始化固定延迟估计器
        
        Args:
            delay_samples: 固定延迟值（采样点数）
        """
        self.delay_samples = max(0, delay_samples)
    
    def estimate_delay(self, mic_signal: np.ndarray, ref_signal: np.ndarray, sample_rate: int) -> int:
        """
        返回固定延迟值（忽略输入信号）
        
        Args:
            mic_signal: 麦克风信号（未使用）
            ref_signal: 参考信号（未使用）
            sample_rate: 采样率（未使用）
            
        Returns:
            预设的固定延迟值
        """
        return self.delay_samples
    
    def get_name(self) -> str:
        return f"Fixed({self.delay_samples} samples)"


def create_delay_estimator(type: str, **kwargs) -> DelayEstimator:
    """
    工厂函数：创建延迟估计器
    
    Args:
        type: 估计器类型，支持 "gcc_phat" 或 "fixed"
        **kwargs: 传递给对应估计器的参数
            - GCCPHATDelayEstimator (type="gcc_phat"):
                - interp: int, 插值倍数，用于提高精度 (默认 1)
                - offset_seconds: float, 偏移量（秒），用于补偿算法误差 (默认 0.001)
            - FixedDelayEstimator (type="fixed"):
                - delay_samples: int, 固定延迟值（采样点数）
        
    Returns:
        DelayEstimator 实例
        
    Raises:
        ValueError: 不支持的估计器类型
    """
    estimator_map = {
        "gcc_phat": GCCPHATDelayEstimator,
        "fixed": FixedDelayEstimator,
    }
    
    if type not in estimator_map:
        supported_types = ", ".join(estimator_map.keys())
        raise ValueError(f"不支持的估计器类型: {type}，支持的类型: {supported_types}")
    
    estimator_class = estimator_map[type]
    return estimator_class(**kwargs)

