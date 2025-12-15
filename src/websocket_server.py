#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebSocket 服务器模块

基于 Tornado 实现 WebSocket 服务器，处理音频数据的双向传输。

协议设计：
- 上行（客户端 → 服务器）：接收二进制 PCM 音频数据，放入 speaker_queue
- 下行（服务器 → 客户端）：从 output_queue 获取处理后的音频，发送给客户端
"""

import asyncio
import queue
import logging
from typing import Optional, Set
import numpy as np
import tornado.web
import tornado.websocket
import tornado.ioloop


logger = logging.getLogger(__name__)


class AECWebSocketHandler(tornado.websocket.WebSocketHandler):
    """
    回声消除 WebSocket 处理器
    
    处理客户端连接、断开、消息接收和发送。
    """
    
    # 类级别的连接集合（用于广播）
    connections: Set['AECWebSocketHandler'] = set()
    
    # 共享的队列引用（在启动时设置）
    speaker_queue: Optional[queue.Queue] = None
    output_queue: Optional[queue.Queue] = None
    
    # 音频配置
    sample_rate: int = 16000
    hop_size: int = 256
    
    def initialize(self):
        """初始化处理器"""
        self._sender_task = None
        self._is_sending = False
    
    def check_origin(self, origin):
        """允许跨域连接"""
        return True
    
    def open(self):
        """WebSocket 连接打开时调用"""
        AECWebSocketHandler.connections.add(self)
        logger.info(f"客户端连接，当前连接数: {len(self.connections)}")
        
        # 启动发送任务
        self._is_sending = True
        self._sender_task = asyncio.create_task(self._send_loop())
    
    def on_close(self):
        """WebSocket 连接关闭时调用"""
        self._is_sending = False
        if self._sender_task:
            self._sender_task.cancel()
        
        AECWebSocketHandler.connections.discard(self)
        logger.info(f"客户端断开，当前连接数: {len(self.connections)}")
    
    def on_message(self, message):
        """
        接收客户端消息
        
        Args:
            message: 二进制 PCM 音频数据（16位整数或32位浮点）
        """
        if self.speaker_queue is None:
            logger.warning("speaker_queue 未初始化")
            return
        
        try:
            # 尝试解析为 float32（首选）
            if len(message) == self.hop_size * 4:
                # 32位浮点格式
                audio_chunk = np.frombuffer(message, dtype=np.float32)
            elif len(message) == self.hop_size * 2:
                # 16位整数格式，转换为浮点
                audio_chunk = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32767.0
            else:
                # 其他长度，尝试自动检测
                if len(message) % 4 == 0:
                    audio_chunk = np.frombuffer(message, dtype=np.float32)
                elif len(message) % 2 == 0:
                    audio_chunk = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32767.0
                else:
                    logger.warning(f"无法解析音频数据，长度: {len(message)}")
                    return
            
            # 放入队列
            try:
                self.speaker_queue.put_nowait(audio_chunk)
            except queue.Full:
                # 队列满时丢弃最旧的数据
                try:
                    self.speaker_queue.get_nowait()
                    self.speaker_queue.put_nowait(audio_chunk)
                except queue.Empty:
                    pass
                    
        except Exception as e:
            logger.error(f"处理音频消息时出错: {e}")
    
    async def _send_loop(self):
        """
        发送循环：从 output_queue 获取处理后的音频并发送
        """
        while self._is_sending:
            try:
                # 非阻塞获取
                if self.output_queue is not None:
                    try:
                        output_chunk = self.output_queue.get_nowait()
                        
                        # 转换为 16 位 PCM 字节
                        pcm_data = (output_chunk * 32767).astype(np.int16).tobytes()
                        
                        # 发送给客户端
                        if self.ws_connection:
                            await self.write_message(pcm_data, binary=True)
                            
                    except queue.Empty:
                        pass
                
                # 短暂休眠，避免空转
                await asyncio.sleep(0.001)
                
            except tornado.websocket.WebSocketClosedError:
                break
            except Exception as e:
                logger.error(f"发送音频时出错: {e}")
                await asyncio.sleep(0.01)
    
    def on_pong(self, data):
        """收到 pong 响应"""
        pass


class HealthCheckHandler(tornado.web.RequestHandler):
    """健康检查处理器"""
    
    def get(self):
        self.write({
            "status": "ok",
            "connections": len(AECWebSocketHandler.connections)
        })


class AECWebSocketServer:
    """
    回声消除 WebSocket 服务器
    
    封装 Tornado 服务器的创建和管理。
    """
    
    def __init__(
        self,
        speaker_queue: queue.Queue,
        output_queue: queue.Queue,
        port: int = 8765,
        sample_rate: int = 16000,
        hop_size: int = 256
    ):
        """
        初始化 WebSocket 服务器
        
        Args:
            speaker_queue: 输入队列（要播放的音频）
            output_queue: 输出队列（处理后的音频）
            port: 监听端口
            sample_rate: 采样率
            hop_size: 每次处理的样本数
        """
        self.speaker_queue = speaker_queue
        self.output_queue = output_queue
        self.port = port
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        
        # 设置处理器的共享资源
        AECWebSocketHandler.speaker_queue = speaker_queue
        AECWebSocketHandler.output_queue = output_queue
        AECWebSocketHandler.sample_rate = sample_rate
        AECWebSocketHandler.hop_size = hop_size
        
        # 创建应用
        self.app = tornado.web.Application([
            (r"/aec", AECWebSocketHandler),
            (r"/health", HealthCheckHandler),
        ])
        
        self._server = None
        self._io_loop = None
    
    def start(self, blocking: bool = True):
        """
        启动服务器
        
        Args:
            blocking: 是否阻塞运行
        """
        self._server = self.app.listen(self.port)
        logger.info(f"WebSocket 服务器启动: ws://0.0.0.0:{self.port}/aec")
        logger.info(f"健康检查端点: http://0.0.0.0:{self.port}/health")
        
        if blocking:
            try:
                tornado.ioloop.IOLoop.current().start()
            except KeyboardInterrupt:
                self.stop()
    
    async def start_async(self):
        """异步启动服务器"""
        self._server = self.app.listen(self.port)
        logger.info(f"WebSocket 服务器启动: ws://0.0.0.0:{self.port}/aec")
        logger.info(f"健康检查端点: http://0.0.0.0:{self.port}/health")
    
    def stop(self):
        """停止服务器"""
        logger.info("正在停止 WebSocket 服务器...")
        
        # 关闭所有连接
        for conn in list(AECWebSocketHandler.connections):
            conn.close()
        
        # 停止服务器
        if self._server:
            self._server.stop()
        
        # 停止 IOLoop - 直接调用 stop()，因为从信号处理器调用时
        # add_callback 可能无法正确执行
        try:
            io_loop = tornado.ioloop.IOLoop.current()
            # if io_loop._running:
            io_loop.stop()
        except Exception as e:
            logger.warning(f"停止 IOLoop 时出错: {e}")
    
    def get_connection_count(self) -> int:
        """获取当前连接数"""
        return len(AECWebSocketHandler.connections)

