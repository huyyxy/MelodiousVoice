import gradio as gr
import websocket
import threading
import time
import numpy as np
import librosa
import socket
import queue

# Constants matching server defaults
SAMPLE_RATE = 16000
CHUNK_SIZE = 256  # Server default hop_size

class AECClient:
    def __init__(self):
        self.ws = None
        self.is_connected = False
        self.is_recording = False
        self.recv_thread = None
        self.recv_buffer = []  # Stores incoming audio chunks when recording
        self.lock = threading.Lock()
        
    def connect(self, server_ip, server_port):
        """Establishes WebSocket connection."""
        if self.is_connected:
            return "Already connected."
            
        uri = f"ws://{server_ip}:{server_port}/aec"
        try:
            self.ws = websocket.create_connection(uri, timeout=5)
            self.is_connected = True
            
            # Start receiving thread
            self.recv_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self.recv_thread.start()
            
            print(f"Connected to {uri}")
            return "Connected successfully."
        except Exception as e:
            return f"Connection failed: {str(e)}"

    def disconnect(self):
        """Closes WebSocket connection."""
        if not self.is_connected:
            return "Not connected."
            
        self.is_connected = False
        if self.ws:
            self.ws.close()
            self.ws = None
        return "Disconnected."

    def _receive_loop(self):
        """Background thread to receive audio from server."""
        while self.is_connected and self.ws:
            try:
                # Use a small timeout to allow checking self.is_connected periodically
                self.ws.sock.settimeout(0.5)
                data = self.ws.recv()
                
                if data:
                    # print(f"Received data: {len(data)} bytes") # Verbose log
                    # Parse binary data
                    # Server sends int16 PCM
                    # We only store it if recording
                    if self.is_recording:
                         processed_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32767.0
                         with self.lock:
                             self.recv_buffer.append(processed_chunk)
                             if len(self.recv_buffer) % 100 == 0:
                                 print(f"Recording buffer size: {len(self.recv_buffer)} chunks")
                             
            except (websocket.WebSocketTimeoutException, socket.timeout):
                continue
            except Exception as e:
                print(f"Receive error: {e}")
                if self.is_connected: # Only print if we didn't intentionally disconnect
                    self.is_connected = False
                break

    def start_recording(self):
        """Starts capturing received audio."""
        if not self.is_connected:
            return "Error: Not connected to server."
        
        with self.lock:
            self.recv_buffer = []
            self.is_recording = True
        return "Recording started..."

    def stop_recording(self):
        """Stops capturing and returns the recorded audio."""
        if not self.is_recording:
            return None, "Error: Not recording."
            
        self.is_recording = False
        
        with self.lock:
            if not self.recv_buffer:
                print("Warning: Recording stopped but buffer is empty.")
                return None, "⚠️ Recording stopped: No data received from server!"
            
            # Concatenate all chunks
            full_audio = np.concatenate(self.recv_buffer)
            
        # Return tuple (sample_rate, data) for Gradio Audio component
        return (SAMPLE_RATE, full_audio), "Recording stopped. Audio ready."

    def upload_and_play(self, audio_path):
        """Reads a file and streams it to the server."""
        if not self.is_connected:
            return "Error: Not connected."
        
        if audio_path is None:
            return "Error: No file selected."

        def send_worker():
            try:
                # Load and resample
                print(f"Loading {audio_path}...")
                y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
                
                # Convert to int16 PCM
                pcm_data = (y * 32767).astype(np.int16)
                
                num_chunks = len(pcm_data) // CHUNK_SIZE
                chunks = [pcm_data[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE] for i in range(num_chunks)]
                if len(pcm_data) % CHUNK_SIZE != 0:
                    chunks.append(pcm_data[num_chunks * CHUNK_SIZE:]) # Last chunk
                
                print(f"Sending {len(chunks)} chunks...")
                
                for chunk in chunks:
                    if not self.is_connected:
                        break
                    
                    try:
                        self.ws.send_binary(chunk.tobytes())
                    except Exception as e:
                        print(f"Send error: {e}")
                        break
                    
                    # Pacing to match real-time
                    time.sleep(CHUNK_SIZE / SAMPLE_RATE)
                    
                print("Upload finished.")
            except Exception as e:
                print(f"Upload/Play error: {e}")

        # Run in a separate thread to not block UI
        threading.Thread(target=send_worker, daemon=True).start()
        return f"Started playing {audio_path}..."

# Global client instance
client = AECClient()

# UI Wrappers
def ui_connect(ip, port):
    return client.connect(ip, port)

def ui_play(file_path):
    return client.upload_and_play(file_path)

def ui_record_start():
    return client.start_recording()

def ui_record_stop():
    return client.stop_recording()

# Gradio Interface
with gr.Blocks(title="NKF-AEC Client") as demo:
    gr.Markdown("# NKF-AEC Web Client v2")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 1. Connection")
            server_ip = gr.Textbox(label="Server IP", value="127.0.0.1")
            server_port = gr.Number(label="Server Port", value=8765, precision=0)
            connect_btn = gr.Button("Connect", variant="primary")
            disconnect_btn = gr.Button("Disconnect")
            connection_status = gr.Label(label="Status", value="Disconnected")
            
        with gr.Column(scale=1):
            gr.Markdown("## 2. Playback (Far-end)")
            input_audio = gr.Audio(label="Upload Audio to Play", type="filepath")
            play_btn = gr.Button("Upload & Play")
            play_status = gr.Label(label="Playback Status", value="Idle")

        with gr.Column(scale=1):
            gr.Markdown("## 3. Recording (Near-end)")
            record_start_btn = gr.Button("Start Recording", variant="stop") # Red mainly for attention
            record_stop_btn = gr.Button("Stop Recording")
            recorded_audio = gr.Audio(label="Recorded Output", type="numpy")
            record_status = gr.Label(label="Recording Status", value="Idle")

    # Event Wiring
    connect_btn.click(
        fn=ui_connect,
        inputs=[server_ip, server_port],
        outputs=[connection_status]
    )
    
    disconnect_btn.click(
        fn=lambda: client.disconnect(),
        outputs=[connection_status]
    )

    play_btn.click(
        fn=ui_play,
        inputs=[input_audio],
        outputs=[play_status]
    )
    
    record_start_btn.click(
        fn=ui_record_start,
        outputs=[record_status]
    )
    
    record_stop_btn.click(
        fn=ui_record_stop,
        outputs=[recorded_audio, record_status]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
