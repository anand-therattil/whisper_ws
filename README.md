# Whisper WebSocket Server (faster-whisper)

A simple WebSocket-based server + client for running **faster-whisper** transcription over WebSockets. The server listens on `localhost` and, by default, uses port **8762**. The client sends an audio file to the server and receives transcription results as JSON messages over the socket.

> This repository assumes you already have Python installed (3.10+ recommended).

---

## Quick start

1. Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

2. Run the server (defaults to `localhost:8762`):

```bash
python whisper_server.py
```

3. Run the client to send an audio file for transcription:

```bash
python client.py path/to/audio.wav
```

The client will open a WebSocket connection to the server, send the audio file, and print received transcription messages.

---

## Files

* `whisper_server.py` — WebSocket server that loads the `faster-whisper` model and accepts audio over WebSocket connections.
* `client.py` — Example client that connects to the server, uploads an audio file, and displays transcription responses.
* `requirements.txt` — Python package list used by `pip install -r requirements.txt`.

---

## Server behavior & configuration

* Default host: `localhost`
* Default port: `8762`

You can change the host/port by editing the top of `whisper_server.py` .

```bash
python whisper_server.py 
```

The server loads a `faster-whisper` model on startup (this can be configured inside `whisper_server.py`). Depending on your model choice and hardware this may take a few seconds to a few minutes on first load.

## Client usage

**Synchronous (send whole file):**

```bash
python client.py path/to/audio.wav
```

## Example logs

**Server startup:**

```
[ASR] Initializing Whisper model...
[ASR] Loading Whisper model: tiny
[2025-12-12 19:44:52.763] [ctranslate2] [thread 291870] [warning] The compute type inferred from the saved model is float16, but the target device or backend do not support efficient float16 computation. The model weights have been automatically converted to use the float32 compute type instead.
[ASR] Model loaded successfully
[ASR] Running model warmup...
[ASR] Model warmup completed
[ASR] Server ready on ws://0.0.0.0:8762
[ASR] Model: tiny, Device: cpu, Audio: 16000Hz 1ch
[ASR] VAD mode: 3, Silence threshold: 1500ms
[ASR] Client connected
[ASR] Processing audio: 34800 samples, 2.17s
[ASR] Transcription completed in 0.37s: 'What services do you offer?...'
[ASR] Client disconnected
```

**Client session:**

```
Whisper WebSocket Client
Audio file: /Users/cmi_10128/Desktop/audio_files/16KHz/general_conversation_0_16k.wav
Server: ws://localhost:8762

Loading audio file: /Users/cmi_10128/Desktop/audio_files/16KHz/general_conversation_0_16k.wav
Audio loaded: 34800.00s, 16000Hz, 69600 bytes
Connecting to ws://localhost:8762...
Server ready, sending audio data...
Sent 40.0 KB...
Audio sent (68.0 KB total)
Waiting for transcription...
Server is processing audio...

============================================================
TRANSCRIPTION RESULT
============================================================
Text: What services do you offer?

Language: en (confidence: 0.98)
Audio duration: 2.17s
Processing time: 0.37s
Real-time factor: 0.17x

Segments (1):
----------------------------------------
 1. [  0.00s -   1.70s] What services do you offer?
============================================================

Result saved to: general_conversation_0_16k_transcription.json
```

---

## Troubleshooting

* **Server won't start** — Verify Python version and that `requirements.txt` dependencies installed successfully.
* **Client can't connect** — Confirm server is running on `localhost:8762` and no firewall is blocking the port. If you changed host/port, make sure the client is using the right WebSocket URL.
* **Audio not recognized / format errors** — Make sure the audio is a supported format (WAV/PCM recommended) and sample rate matches the metadata you send.


---
If you want, I can also generate a `requirements.txt` example and a minimal `client.py` / `whisper_server.py` skeleton that implements the protocol described above.
