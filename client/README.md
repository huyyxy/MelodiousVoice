# NKF-AEC Web Client v2

This is a web-based client for the NKF-AEC Real-time Echo Cancellation Server.
It uses Gradio for the UI and `websocket-client` to communicate with the server.

## Features

- **Persistent Connection**: connect once and keep the session alive.
- **Playback (Far-end)**: Upload a WAV file to send to the server for playback.
- **Recording (Near-end)**: Real-time capture of the echo-cancelled microphone signal from the server.
- **Configurable**: Easily change the server IP and port from the UI.

## Installation

1.  Navigate to the `client` directory:
    ```bash
    cd client
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Ensure the **AEC Server** is running on your target machine:
    ```bash
    # On the server machine
    python3 src/main.py --model nkf_epoch70.pt --port 8765
    ```

2.  Start the **Web Client**:
    ```bash
    # On this machine
    python3 app.py
    ```

3.  Open the browser using the URL shown in the terminal (usually `http://0.0.0.0:7860`).

4.  **Workflow**:
    - **Step 1: Connect**: Enter Server IP/Port and click **Connect**. Ensure status says "Connected successfully".
    - **Step 2: Playback**: Upload an audio file in the "Playback" section and click **Upload & Play**. The server will play this audio.
    - **Step 3: Record**: Click **Start Recording** in the "Recording" section. Speak or let the server capture audio. Click **Stop Recording** when done.
    - **Step 4: Listen**: The recorded (echo-cancelled) audio will appear in the "Recorded Output" player.

## Notes

- The client resamples input audio to **16000Hz** before sending.
- Recording happens in memory; very long recordings might consume significant RAM.
- Ensure network connectivity between client and server (check firewalls if connection fails).
