#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKF-AEC WebSocket 实时回声消除系统 - 主程序入口

功能说明：
1. 启动 WebSocket 服务器接收客户端音频
2. 启动音频处理线程进行回声消除
3. 将处理后的音频发送回客户端

用法:
    python3 main.py --list-devices
    python3 main.py --model nkf_epoch70.pt --port 8765
    python3 main.py --model nkf_epoch70.pt --fixed-delay 1600 --mic-device 0 --speaker-device 1
"""

import argparse
import asyncio
import logging
import os
import queue
import signal
import sys
import threading
import time

import numpy as np
import tornado.ioloop

from delay_estimator import create_delay_estimator, GCCPHATDelayEstimator
from aec_processor import create_aec_processor
from audio_manager import AudioManager
from audio_processing_thread import AudioProcessingThread, AlignmentHelper
from websocket_server import AECWebSocketServer


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RealtimeAECServer:
    """
    实时回声消除服务器
    
    整合所有组件，提供完整的实时回声消除服务。
    """
    
    def __init__(
        self,
        model_path: str,
        port: int = 8765,
        sample_rate: int = 16000,
        hop_size: int = 256,
        block_size: int = 1024,
        mic_device_idx: int = None,
        speaker_device_idx: int = None,
        fixed_delay: int = None,
        align_duration: float = 2.0,
        max_delay: int = 4800,
        queue_size: int = 100,
        align_file: str = None
    ):
        """
        初始化实时回声消除服务器
        
        Args:
            model_path: NKF 模型权重文件路径
            port: WebSocket 服务器端口
            sample_rate: 采样率 (Hz)
            hop_size: 每次处理的样本数
            block_size: STFT 窗口大小
            mic_device_idx: 麦克风设备索引（None 表示使用默认设备）
            speaker_device_idx: 扬声器设备索引（None 表示使用默认设备）
            fixed_delay: 固定延迟值（采样点），如果指定则跳过自动对齐
            align_duration: 对齐阶段时长（秒）
            max_delay: 最大延迟（采样点数）
            queue_size: 队列最大长度
            align_file: 对齐使用的音频文件路径
        """
        self.model_path = model_path
        self.port = port
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.block_size = block_size
        self.mic_device_idx = mic_device_idx
        self.speaker_device_idx = speaker_device_idx
        self.fixed_delay = fixed_delay
        self.align_duration = align_duration
        self.max_delay = max_delay
        self.align_file = align_file
        
        # 创建线程安全队列
        self.speaker_queue = queue.Queue(maxsize=queue_size)
        self.output_queue = queue.Queue(maxsize=queue_size)
        
        # 组件占位符
        self.delay_estimator = None
        self.aec_processor = None
        self.audio_manager = None
        self.audio_thread = None
        self.websocket_server = None
        
        # 状态
        self.delay_samples = 0
        self._running = False
        self._shutdown_event = threading.Event()
    
    def _resolve_model_path(self, model_path: str) -> str:
        """解析模型路径"""
        if os.path.isabs(model_path):
            return model_path
        
        if os.path.exists(model_path):
            return model_path
        
        # 尝试相对于脚本目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path_in_script_dir = os.path.join(script_dir, model_path)
        if os.path.exists(model_path_in_script_dir):
            return model_path_in_script_dir
        
        return model_path
    
    def initialize(self):
        """初始化所有组件"""
        logger.info("正在初始化组件...")
        
        # 1. 创建延迟估计器
        logger.info("创建延迟估计器...")
        self.delay_estimator = create_delay_estimator(
            fixed_delay=self.fixed_delay,
            interp=1,
            offset_seconds=0.001
        )
        logger.info(f"延迟估计器: {self.delay_estimator.get_name()}")
        
        # 2. 创建 AEC 处理器
        logger.info("加载 NKF 模型...")
        model_path = self._resolve_model_path(self.model_path)
        self.aec_processor = create_aec_processor(
            model_path=model_path,
            block_size=self.block_size,
            hop_size=self.hop_size,
            device='cpu'
        )
        logger.info(f"AEC 处理器: {self.aec_processor.get_name()}")
        
        # 3. 创建音频管理器
        logger.info("初始化音频管理器...")
        self.audio_manager = AudioManager(
            sample_rate=self.sample_rate,
            channels=1,
            hop_size=self.hop_size,
            mic_device_idx=self.mic_device_idx,
            speaker_device_idx=self.speaker_device_idx
        )
        logger.info(f"音频设备: {self.audio_manager.get_device_info_str()}")
        
        # 4. 执行初始对齐（如果未指定固定延迟）
        if self.fixed_delay is not None:
            self.delay_samples = self.fixed_delay
            logger.info(f"使用固定延迟值: {self.delay_samples} 采样点 "
                       f"({self.delay_samples / self.sample_rate * 1000:.1f} ms)")
        elif self.align_duration > 0:
            logger.info("执行初始对齐...")
            alignment_helper = AlignmentHelper(
                audio_manager=self.audio_manager,
                delay_estimator=self.delay_estimator,
                sample_rate=self.sample_rate,
                hop_size=self.hop_size,
                align_duration=self.align_duration,
                align_file=self.align_file
            )
            self.delay_samples = alignment_helper.run_alignment()
            
            # 重置 AEC 处理器状态
            self.aec_processor.reset_state()
        else:
            self.delay_samples = 0
            logger.info("跳过对齐阶段，使用默认延迟 0")
        
        # 5. 创建音频处理线程
        logger.info("创建音频处理线程...")
        self.audio_thread = AudioProcessingThread(
            audio_manager=self.audio_manager,
            aec_processor=self.aec_processor,
            speaker_queue=self.speaker_queue,
            output_queue=self.output_queue,
            delay_samples=self.delay_samples,
            max_delay=self.max_delay,
            hop_size=self.hop_size,
            sample_rate=self.sample_rate
        )
        
        # 6. 创建 WebSocket 服务器
        logger.info("创建 WebSocket 服务器...")
        self.websocket_server = AECWebSocketServer(
            speaker_queue=self.speaker_queue,
            output_queue=self.output_queue,
            port=self.port,
            sample_rate=self.sample_rate,
            hop_size=self.hop_size
        )
        
        logger.info("初始化完成!")
    
    def run(self):
        """运行服务器"""
        self._running = True
        self._stop_count = 0
        
        # 设置信号处理
        def signal_handler(sig, frame):
            self._stop_count += 1
            if self._stop_count == 1:
                logger.info("\n收到中断信号，正在停止...")
                self.stop()
            elif self._stop_count >= 2:
                # 第二次 Ctrl+C 强制退出
                logger.info("\n再次收到中断信号，强制退出...")
                import os
                os._exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            # 启动音频处理线程
            logger.info("启动音频处理线程...")
            self.audio_thread.start()
            
            # 启动 WebSocket 服务器（阻塞）
            logger.info(f"启动 WebSocket 服务器，端口: {self.port}")
            print("\n" + "=" * 60)
            print(f"  NKF-AEC 实时回声消除服务器已启动")
            print(f"  WebSocket 端点: ws://0.0.0.0:{self.port}/aec")
            print(f"  健康检查: http://0.0.0.0:{self.port}/health")
            print("=" * 60)
            print("\n按 Ctrl+C 停止服务器\n")
            
            self.websocket_server.start(blocking=True)
            
        except Exception as e:
            logger.error(f"服务器错误: {e}")
        finally:
            self.stop()
    
    async def run_async(self):
        """异步运行服务器"""
        self._running = True
        
        try:
            # 启动音频处理线程
            logger.info("启动音频处理线程...")
            self.audio_thread.start()
            
            # 异步启动 WebSocket 服务器
            await self.websocket_server.start_async()
            
            logger.info(f"服务器已启动，端口: {self.port}")
            
            # 等待停止信号
            while self._running and not self._shutdown_event.is_set():
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error(f"服务器错误: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """停止服务器"""
        if not self._running:
            return
        
        self._running = False
        self._shutdown_event.set()
        
        logger.info("正在停止服务器...")
        
        # 停止音频处理线程
        if self.audio_thread:
            self.audio_thread.stop()
            self.audio_thread.join(timeout=2.0)
        
        # 停止 WebSocket 服务器
        if self.websocket_server:
            self.websocket_server.stop()
        
        # 清理音频管理器
        if self.audio_manager:
            self.audio_manager.terminate()
        
        logger.info("服务器已停止")
    
    def get_stats(self) -> dict:
        """获取运行统计"""
        stats = {
            'running': self._running,
            'delay_samples': self.delay_samples,
            'delay_ms': self.delay_samples / self.sample_rate * 1000,
            'speaker_queue_size': self.speaker_queue.qsize(),
            'output_queue_size': self.output_queue.qsize(),
        }
        
        if self.audio_thread:
            stats.update(self.audio_thread.get_stats())
        
        if self.websocket_server:
            stats['connections'] = self.websocket_server.get_connection_count()
        
        return stats


def list_audio_devices():
    """列出所有可用的音频设备"""
    audio_manager = AudioManager()
    audio_manager.print_devices()
    audio_manager.terminate()


def main():
    parser = argparse.ArgumentParser(
        description="NKF-AEC WebSocket 实时回声消除系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 列出所有音频设备
  python3 main.py --list-devices
  
  # 启动服务器（使用系统默认设备）
  python3 main.py --model nkf_epoch70.pt
  
  # 指定端口和固定延迟
  python3 main.py --model nkf_epoch70.pt --port 8765 --fixed-delay 1600
  
  # 指定音频设备
  python3 main.py --model nkf_epoch70.pt --mic-device 0 --speaker-device 1

客户端连接示例（Python）:
  import websockets
  import asyncio
  import numpy as np
  
  async def client():
      uri = "ws://localhost:8765/aec"
      async with websockets.connect(uri) as ws:
          # 发送音频
          audio = np.random.randn(256).astype(np.float32)
          pcm = (audio * 32767).astype(np.int16).tobytes()
          await ws.send(pcm)
          
          # 接收处理后的音频
          response = await ws.recv()
          processed = np.frombuffer(response, dtype=np.int16) / 32767.0
  
  asyncio.run(client())
        """
    )
    
    parser.add_argument('--list-devices', action='store_true',
                        help='列出所有可用的音频设备')
    parser.add_argument('--model', type=str, default='nkf_epoch70.pt',
                        help='NKF 模型权重文件路径')
    parser.add_argument('--port', type=int, default=8765,
                        help='WebSocket 服务器端口（默认: 8765）')
    parser.add_argument('--sample-rate', type=int, default=16000,
                        help='采样率 (默认: 16000)')
    parser.add_argument('--hop-size', type=int, default=256,
                        help='每次处理的样本数 (默认: 256)')
    parser.add_argument('--block-size', type=int, default=1024,
                        help='STFT 窗口大小 (默认: 1024)')
    parser.add_argument('--mic-device', type=int, default=None,
                        help='麦克风设备索引（不指定则使用系统默认）')
    parser.add_argument('--speaker-device', type=int, default=None,
                        help='扬声器设备索引（不指定则使用系统默认）')
    parser.add_argument('--fixed-delay', type=int, default=None,
                        help='固定延迟值（采样点），跳过自动对齐')
    parser.add_argument('--align-duration', type=float, default=2.0,
                        help='对齐阶段时长，单位秒 (默认: 2.0)')
    parser.add_argument('--max-delay', type=int, default=4800,
                        help='最大延迟（采样点数，默认: 4800，约 300ms）')
    parser.add_argument('--queue-size', type=int, default=10000,
                        help='队列最大长度 (默认: 10000)')
    parser.add_argument('--align-file', type=str, default=None,
                        help='用于对齐的参考音频文件路径 (默认: 使用白噪声)')
    
    args = parser.parse_args()
    
    if args.list_devices:
        list_audio_devices()
        return
    
    # 创建并运行服务器
    server = RealtimeAECServer(
        model_path=args.model,
        port=args.port,
        sample_rate=args.sample_rate,
        hop_size=args.hop_size,
        block_size=args.block_size,
        mic_device_idx=args.mic_device,
        speaker_device_idx=args.speaker_device,
        fixed_delay=args.fixed_delay,
        align_duration=args.align_duration,
        max_delay=args.max_delay,
        queue_size=args.queue_size,
        align_file=args.align_file
    )
    
    try:
        server.initialize()
        server.run()
    except FileNotFoundError as e:
        logger.error(f"文件未找到: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

