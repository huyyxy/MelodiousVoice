#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频处理线程模块

独立线程运行音频处理循环，处理麦克风输入和扬声器输出，
并调用 AEC 处理器进行回声消除。
"""

import threading
import queue
import time
from typing import Optional
import numpy as np
import soundfile as sf
import os
from scipy import signal as scipy_signal

from audio_manager import AudioManager
from aec_processor import AECProcessor
from delay_estimator import DelayEstimator


class ReferenceBuffer:
    """
    参考信号缓冲区
    
    用于存储播放的参考信号，支持延迟补偿访问。
    """
    
    def __init__(self, max_delay: int, hop_size: int):
        """
        初始化参考信号缓冲区
        
        Args:
            max_delay: 最大延迟（采样点数）
            hop_size: 每次处理的样本数
        """
        self.max_delay = max_delay
        self.hop_size = hop_size
        # 缓冲区长度 = 最大延迟 + 额外缓冲
        self.buffer_size = max_delay + hop_size * 10
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.write_idx = 0
    
    def push(self, chunk: np.ndarray):
        """
        推入一个音频块
        
        Args:
            chunk: 音频块 (numpy array)
        """
        chunk_len = len(chunk)
        
        # 写入缓冲区（环形写入）
        end_idx = self.write_idx + chunk_len
        if end_idx <= self.buffer_size:
            self.buffer[self.write_idx:end_idx] = chunk
        else:
            # 跨越边界
            first_part = self.buffer_size - self.write_idx
            self.buffer[self.write_idx:] = chunk[:first_part]
            self.buffer[:chunk_len - first_part] = chunk[first_part:]
        
        self.write_idx = end_idx % self.buffer_size
    
    def get_delayed(self, delay: int) -> np.ndarray:
        """
        获取延迟补偿后的音频块
        
        Args:
            delay: 延迟采样点数
            
        Returns:
            延迟补偿后的音频块
        """
        # 计算读取位置（相对于写入位置向后偏移）
        read_start = (self.write_idx - delay - self.hop_size + self.buffer_size) % self.buffer_size
        read_end = read_start + self.hop_size
        
        if read_end <= self.buffer_size:
            return self.buffer[read_start:read_end].copy()
        else:
            # 跨越边界
            first_part = self.buffer_size - read_start
            return np.concatenate([
                self.buffer[read_start:],
                self.buffer[:self.hop_size - first_part]
            ])
    
    def reset(self):
        """重置缓冲区"""
        self.buffer.fill(0)
        self.write_idx = 0


class AudioProcessingThread(threading.Thread):
    """
    音频处理线程
    
    独立线程运行音频处理循环：
    1. 从 speaker_queue 获取要播放的音频
    2. 播放音频到扬声器
    3. 录制麦克风音频
    4. 延迟补偿：获取对应的参考音频
    5. AEC 处理：调用 aec_processor.process_chunk()
    6. 将处理后的音频放入 output_queue
    """
    
    def __init__(
        self,
        audio_manager: AudioManager,
        aec_processor: AECProcessor,
        speaker_queue: queue.Queue,
        output_queue: queue.Queue,
        delay_samples: int = 0,
        max_delay: int = 4800,
        hop_size: int = 256,
        sample_rate: int = 16000
    ):
        """
        初始化音频处理线程
        
        Args:
            audio_manager: 音频管理器
            aec_processor: AEC 处理器
            speaker_queue: 输入队列（要播放的音频）
            output_queue: 输出队列（处理后的音频）
            delay_samples: 延迟采样点数
            max_delay: 最大延迟（采样点数）
            hop_size: 每次处理的样本数
            sample_rate: 采样率
        """
        super().__init__(daemon=True)
        
        self.audio_manager = audio_manager
        self.aec_processor = aec_processor
        self.speaker_queue = speaker_queue
        self.output_queue = output_queue
        self.delay_samples = delay_samples
        self.hop_size = hop_size
        self.sample_rate = sample_rate
        
        # 参考信号缓冲区
        self.ref_buffer = ReferenceBuffer(max_delay=max_delay, hop_size=hop_size)
        
        # 静音块（当队列为空时使用）
        self.silence_chunk = np.zeros(hop_size, dtype=np.float32)
        
        # 控制事件
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # 默认不暂停
        
        # 统计信息
        self.processed_chunks = 0
        self.start_time = 0
    
    def set_delay(self, delay_samples: int):
        """设置延迟值"""
        self.delay_samples = max(0, delay_samples)
    
    def stop(self):
        """停止线程"""
        self._stop_event.set()
    
    def pause(self):
        """暂停处理"""
        self._pause_event.clear()
    
    def resume(self):
        """恢复处理"""
        self._pause_event.set()
    
    def is_stopped(self) -> bool:
        """检查是否已停止"""
        return self._stop_event.is_set()
    
    def run(self):
        """线程主循环"""
        self.start_time = time.time()
        self.processed_chunks = 0
        
        print("音频处理线程启动...")
        
        try:
            # 打开音频流
            self.audio_manager.open_stream()
            
            while not self._stop_event.is_set():
                # 等待恢复（如果暂停）
                self._pause_event.wait()
                
                # 1. 从队列获取要播放的音频（超时则使用静音）
                try:
                    speaker_chunk = self.speaker_queue.get(timeout=0.001)
                except queue.Empty:
                    speaker_chunk = self.silence_chunk.copy()
                
                # 确保格式正确
                if not isinstance(speaker_chunk, np.ndarray):
                    speaker_chunk = np.frombuffer(speaker_chunk, dtype=np.float32)
                
                # 确保长度正确
                if len(speaker_chunk) != self.hop_size:
                    if len(speaker_chunk) < self.hop_size:
                        speaker_chunk = np.pad(speaker_chunk, (0, self.hop_size - len(speaker_chunk)))
                    else:
                        speaker_chunk = speaker_chunk[:self.hop_size]
                
                # 2. 推入参考信号缓冲区
                self.ref_buffer.push(speaker_chunk)
                
                # 3. 同步播放和录制
                mic_chunk = self.audio_manager.read_write_chunk(speaker_chunk)
                
                # 4. 获取延迟补偿后的参考信号
                ref_chunk = self.ref_buffer.get_delayed(self.delay_samples)
                
                # 5. AEC 处理
                try:
                    output_chunk = self.aec_processor.process_chunk(mic_chunk, ref_chunk)
                except Exception as e:
                    print(f"AEC 处理错误: {e}")
                    output_chunk = mic_chunk.copy()
                
                # 6. 将处理后的音频放入输出队列
                try:
                    self.output_queue.put_nowait(output_chunk)
                except queue.Full:
                    # 队列满时丢弃最旧的数据
                    try:
                        self.output_queue.get_nowait()
                        self.output_queue.put_nowait(output_chunk)
                    except queue.Empty:
                        pass
                
                self.processed_chunks += 1
                
        except Exception as e:
            print(f"音频处理线程错误: {e}")
        
        finally:
            # 关闭音频流
            self.audio_manager.close_stream()
            print("音频处理线程结束")
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        elapsed = time.time() - self.start_time if self.start_time > 0 else 0
        processed_seconds = self.processed_chunks * self.hop_size / self.sample_rate
        
        return {
            'processed_chunks': self.processed_chunks,
            'processed_seconds': processed_seconds,
            'elapsed_seconds': elapsed,
            'realtime_ratio': processed_seconds / elapsed if elapsed > 0 else 0
        }


class AlignmentHelper:
    """
    对齐辅助类
    
    用于初始对齐阶段，计算系统延迟。
    """
    
    def __init__(
        self,
        audio_manager: AudioManager,
        delay_estimator: DelayEstimator,
        sample_rate: int = 16000,
        hop_size: int = 256,
        align_duration: float = 2.0,
        align_file: str = None
    ):
        """
        初始化对齐辅助类
        
        Args:
            audio_manager: 音频管理器
            delay_estimator: 延迟估计器
            sample_rate: 采样率
            hop_size: 每次处理的样本数
            align_duration: 对齐持续时间（秒）
        """
        self.audio_manager = audio_manager
        self.delay_estimator = delay_estimator
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.align_duration = align_duration
        self.align_file = align_file
    
    def _load_align_file(self) -> np.ndarray:
        """加载并预处理对齐音频文件"""
        if not os.path.exists(self.align_file):
            print(f"警告: 对齐文件未找到: {self.align_file}，将使用白噪声")
            return None
        
        try:
            audio, sr = sf.read(self.align_file, dtype='float32')
            
            # 转换为单声道
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            
            # 重采样
            if sr != self.sample_rate:
                print(f"对齐文件重采样: {sr} Hz -> {self.sample_rate} Hz")
                num_samples = int(len(audio) * self.sample_rate / sr)
                audio = scipy_signal.resample(audio, num_samples).astype(np.float32)
            
            return audio.astype(np.float32)
        except Exception as e:
            print(f"加载对齐文件失败: {e}，将使用白噪声")
            return None
    
    def generate_test_signal(self, duration: float) -> np.ndarray:
        """
        生成测试信号（白噪声）
        
        Args:
            duration: 持续时间（秒）
            
        Returns:
            测试信号
        """
        num_samples = int(duration * self.sample_rate)
        # 生成白噪声的幅度 越高则信噪比越高
        white_noise_amplitude = 0.5
        return (np.random.randn(num_samples) * white_noise_amplitude).astype(np.float32)
    
    def run_alignment(self) -> int:
        """
        运行对齐流程
        
        Returns:
            检测到的延迟（采样点数）
        """
        print(f"开始对齐阶段 ({self.align_duration}秒)...")
        print("请确保扬声器有足够音量，让麦克风能够捕获到回声...")
        
        # 准备测试信号
        test_signal = None
        if self.align_file:
            print(f"使用音频文件进行对齐: {self.align_file}")
            test_signal = self._load_align_file()
            
            # 如果文件比 align_duration 短，可能需要循环或补零，或者截取
            # 这里简单起见，如果加载成功，使用文件的长度
            if test_signal is not None:
                # 确保长度至少为 align_duration，如果不够则循环
                min_samples = int(self.align_duration * self.sample_rate)
                if len(test_signal) < min_samples:
                     repeats = min_samples // len(test_signal) + 1
                     test_signal = np.tile(test_signal, repeats)
                
                # 截取到指定时长（或者就用文件长度？）
                # 为了保持逻辑一致性，截取到 align_duration
                # align_samples = int(self.align_duration * self.sample_rate)
                # test_signal = test_signal[:align_samples]
                pass
        
        if test_signal is None:
            print("使用白噪声进行对齐 (振幅 0.5)...")
            test_signal = self.generate_test_signal(self.align_duration)
        
        num_chunks = len(test_signal) // self.hop_size
        
        # 录制的音频
        recorded_frames = []
        
        # 打开音频流
        self.audio_manager.open_stream()
        
        try:
            for i in range(num_chunks):
                # 获取当前播放块
                out_chunk = test_signal[i * self.hop_size:(i + 1) * self.hop_size]
                
                # 同步播放和录制
                in_chunk = self.audio_manager.read_write_chunk(out_chunk)
                recorded_frames.append(in_chunk)
                
        finally:
            self.audio_manager.close_stream()
        
        # 合并录音数据
        recorded_signal = np.concatenate(recorded_frames)
        ref_signal = test_signal[:len(recorded_signal)]
        
        # 估计延迟
        delay = self.delay_estimator.estimate_delay(
            recorded_signal, ref_signal, self.sample_rate
        )
        
        print(f"检测到延迟: {delay} 采样点 ({delay / self.sample_rate * 1000:.1f} ms)")
        
        return delay

