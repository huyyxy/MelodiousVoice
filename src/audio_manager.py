#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频管理器模块

封装 PyAudio 的初始化和流管理，提供线程安全的音频输入输出接口。
"""

import threading
from typing import Optional, List, Dict, Any
import numpy as np
import pyaudio


class AudioDevice:
    """音频设备信息"""
    
    def __init__(self, index: int, info: Dict[str, Any]):
        self.index = index
        self.name = info.get('name', 'Unknown')
        self.max_input_channels = info.get('maxInputChannels', 0)
        self.max_output_channels = info.get('maxOutputChannels', 0)
        self.default_sample_rate = int(info.get('defaultSampleRate', 16000))
    
    @property
    def is_input(self) -> bool:
        return self.max_input_channels > 0
    
    @property
    def is_output(self) -> bool:
        return self.max_output_channels > 0
    
    @property
    def device_type(self) -> str:
        if self.is_input and self.is_output:
            return "输入/输出"
        elif self.is_input:
            return "输入"
        elif self.is_output:
            return "输出"
        else:
            return "未知"
    
    def __repr__(self):
        return f"AudioDevice({self.index}: {self.name}, type={self.device_type})"


class AudioManager:
    """
    音频管理器
    
    封装 PyAudio 的初始化和流管理，提供：
    - 设备列表查询
    - 麦克风输入流管理
    - 扬声器输出流管理
    - 线程安全的读写接口
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        hop_size: int = 256,
        mic_device_idx: Optional[int] = None,
        speaker_device_idx: Optional[int] = None,
        audio_format: int = pyaudio.paFloat32
    ):
        """
        初始化音频管理器
        
        Args:
            sample_rate: 采样率（Hz）
            channels: 声道数
            hop_size: 每次处理的样本数
            mic_device_idx: 麦克风设备索引（None 表示使用默认设备）
            speaker_device_idx: 扬声器设备索引（None 表示使用默认设备）
            audio_format: 音频格式（默认 paFloat32）
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.hop_size = hop_size
        self.mic_device_idx = mic_device_idx
        self.speaker_device_idx = speaker_device_idx
        self.audio_format = audio_format
        
        # 初始化 PyAudio
        self.pa = pyaudio.PyAudio()
        
        # 流对象
        self.stream: Optional[pyaudio.Stream] = None
        
        # 线程锁
        self._lock = threading.Lock()
        
        # 状态标志
        self._is_running = False
    
    def list_devices(self) -> List[AudioDevice]:
        """
        列出所有可用的音频设备
        
        Returns:
            AudioDevice 列表
        """
        devices = []
        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            devices.append(AudioDevice(i, info))
        return devices
    
    def print_devices(self):
        """打印所有可用的音频设备"""
        print("\n可用的音频设备:")
        print("-" * 80)
        
        for device in self.list_devices():
            print(f"[{device.index:2d}] {device.name}")
            print(f"     类型: {device.device_type}, "
                  f"输入通道: {device.max_input_channels}, "
                  f"输出通道: {device.max_output_channels}, "
                  f"采样率: {device.default_sample_rate} Hz")
        
        print("-" * 80)
        print("\n提示:")
        print("  - 不指定设备时，程序将使用系统默认的输入/输出设备")
        print("  - 可以用 --mic-device 和 --speaker-device 分别指定麦克风和扬声器\n")
    
    def get_default_input_device(self) -> Optional[int]:
        """获取默认输入设备索引"""
        try:
            info = self.pa.get_default_input_device_info()
            return info.get('index')
        except IOError:
            return None
    
    def get_default_output_device(self) -> Optional[int]:
        """获取默认输出设备索引"""
        try:
            info = self.pa.get_default_output_device_info()
            return info.get('index')
        except IOError:
            return None
    
    def open_stream(self) -> pyaudio.Stream:
        """
        打开音频流（同时支持输入和输出）
        
        Returns:
            PyAudio Stream 对象
        """
        with self._lock:
            if self.stream is not None:
                self.close_stream()
            
            # 构建流参数
            stream_kwargs = {
                'format': self.audio_format,
                'channels': self.channels,
                'rate': self.sample_rate,
                'input': True,
                'output': True,
                'frames_per_buffer': self.hop_size
            }
            
            # 只有在明确指定设备时才添加设备索引
            if self.mic_device_idx is not None:
                stream_kwargs['input_device_index'] = self.mic_device_idx
            if self.speaker_device_idx is not None:
                stream_kwargs['output_device_index'] = self.speaker_device_idx
            
            self.stream = self.pa.open(**stream_kwargs)
            self._is_running = True
            
            return self.stream
    
    def close_stream(self):
        """关闭音频流"""
        with self._lock:
            if self.stream is not None:
                try:
                    if self.stream.is_active():
                        self.stream.stop_stream()
                    self.stream.close()
                except Exception as e:
                    print(f"关闭音频流时出错: {e}")
                finally:
                    self.stream = None
                    self._is_running = False
    
    def read_chunk(self) -> np.ndarray:
        """
        从麦克风读取一个音频块
        
        Returns:
            音频数据 (numpy array, float32)
        """
        if self.stream is None:
            raise RuntimeError("音频读取流未打开")
        
        in_bytes = self.stream.read(self.hop_size, exception_on_overflow=False)
        return np.frombuffer(in_bytes, dtype=np.float32)
    
    def write_chunk(self, chunk: np.ndarray):
        """
        写入一个音频块到扬声器
        
        Args:
            chunk: 音频数据 (numpy array)
        """
        if self.stream is None:
            raise RuntimeError("音频写入流未打开")
        
        # 确保是 float32 格式
        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)
        
        # 确保长度正确
        if len(chunk) < self.hop_size:
            chunk = np.pad(chunk, (0, self.hop_size - len(chunk)))
        elif len(chunk) > self.hop_size:
            chunk = chunk[:self.hop_size]
        
        self.stream.write(chunk.tobytes())
    
    def read_write_chunk(self, output_chunk: np.ndarray) -> np.ndarray:
        """
        同步读写：播放输出并录制输入
        
        Args:
            output_chunk: 要播放的音频数据
            
        Returns:
            录制的音频数据
        """
        self.write_chunk(output_chunk)
        return self.read_chunk()
    
    @property
    def is_running(self) -> bool:
        """检查音频流是否正在运行"""
        return self._is_running and self.stream is not None
    
    def terminate(self):
        """释放 PyAudio 资源"""
        self.close_stream()
        if self.pa is not None:
            try:
                self.pa.terminate()
            except Exception as e:
                print(f"终止 PyAudio 时出错: {e}")
            finally:
                self.pa = None
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.terminate()
        return False
    
    def get_device_info_str(self) -> str:
        """获取当前设备信息字符串"""
        if self.mic_device_idx is None and self.speaker_device_idx is None:
            return "使用系统默认输入/输出设备"
        
        parts = []
        if self.mic_device_idx is not None:
            parts.append(f"麦克风设备: {self.mic_device_idx}")
        else:
            parts.append("麦克风: 系统默认")
        
        if self.speaker_device_idx is not None:
            parts.append(f"扬声器设备: {self.speaker_device_idx}")
        else:
            parts.append("扬声器: 系统默认")
        
        return ", ".join(parts)

