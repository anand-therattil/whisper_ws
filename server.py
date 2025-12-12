#!/usr/bin/env python3
# whisper_ws_server.py
# -*- coding: utf-8 -*-

import os
import time
import asyncio
import json
import uuid
import threading
import queue
import io
from typing import Optional, Dict, Any, List
from contextlib import suppress
import numpy as np

from faster_whisper import WhisperModel
import webrtcvad
from websockets.asyncio.server import serve, ServerConnection
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK, ConnectionClosedError

# Configuration from environment variables
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "tiny")  # tiny, base, small, medium, large-v3
DEVICE = os.environ.get("DEVICE", "cpu")  # cpu, cuda, auto
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "default")  # default, float16, int8, etc.
CPU_THREADS = int(os.environ.get("CPU_THREADS", "0"))  # 0 for auto

# Audio settings
SAMPLE_RATE = int(os.environ.get("SAMPLE_RATE", "16000"))  # Whisper expects 16kHz
CHANNELS = int(os.environ.get("CHANNELS", "1"))  # Mono
SAMPLE_WIDTH = int(os.environ.get("SAMPLE_WIDTH", "2"))  # 16-bit

# Processing settings
VAD_MODE = int(os.environ.get("VAD_MODE", "3"))  # 0-3, 3 is most aggressive
SILENCE_THRESHOLD_MS = int(os.environ.get("SILENCE_THRESHOLD_MS", "1500"))
MIN_AUDIO_LENGTH_MS = int(os.environ.get("MIN_AUDIO_LENGTH_MS", "100"))
MAX_AUDIO_LENGTH_MS = int(os.environ.get("MAX_AUDIO_LENGTH_MS", "30000"))

# Whisper parameters
LANGUAGE = os.environ.get("LANGUAGE", None)  # None for auto-detect
TASK = os.environ.get("TASK", "transcribe")  # transcribe or translate
BEAM_SIZE = int(os.environ.get("BEAM_SIZE", "5"))
BEST_OF = int(os.environ.get("BEST_OF", "5"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.0"))

# Connection settings
REQ_TIMEOUT_S = float(os.environ.get("REQ_TIMEOUT_S", "60"))
CONCURRENCY = int(os.environ.get("CONCURRENCY", "4"))
PERSISTENT_WS = os.environ.get("PERSISTENT_WS", "1") != "0"

# Global model and semaphore
_model: Optional[WhisperModel] = None
_vad: Optional[webrtcvad.Vad] = None
_sem = asyncio.Semaphore(CONCURRENCY)

def init_model_once():
    """Initialize Whisper model and VAD detector"""
    global _model, _vad
    if _model:
        return
    
    print(f"[ASR] Loading Whisper model: {MODEL_SIZE}")
    
    # Prepare WhisperModel arguments
    model_kwargs = {
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE if COMPUTE_TYPE != "auto" else "default",
    }
    
    # Only add cpu_threads if it's specified and device is CPU
    if CPU_THREADS > 0 and DEVICE.lower() in ["cpu", "auto"]:
        model_kwargs["cpu_threads"] = CPU_THREADS
    
    try:
        _model = WhisperModel(MODEL_SIZE, **model_kwargs)
    except Exception as e:
        print(f"[ASR] Error loading model with {model_kwargs}: {e}")
        # Fallback to minimal configuration
        print("[ASR] Trying fallback configuration...")
        _model = WhisperModel(MODEL_SIZE, device=DEVICE)
    
    # Initialize VAD
    _vad = webrtcvad.Vad(VAD_MODE)
    print(f"[ASR] Model loaded successfully")

async def warmup_model():
    """Warm up the model with a short audio clip"""
    if not _model:
        return
    
    # Generate a short silence for warmup
    duration_samples = SAMPLE_RATE  # 1 second
    dummy_audio = np.zeros(duration_samples, dtype=np.float32)
    
    try:
        # Run warmup in executor since transcribe is sync
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: list(_model.transcribe(
                dummy_audio,
                language=LANGUAGE,
                task=TASK,
                beam_size=1,  # Fast for warmup
                best_of=1,
                temperature=TEMPERATURE
            )[0])  # [0] gets segments, consume generator
        )
        print("[ASR] Model warmup completed")
    except Exception as e:
        print(f"[ASR] Warmup failed: {e}")

def transcribe_audio_sync(audio_data: np.ndarray) -> Dict[str, Any]:
    """Transcribe audio using Whisper (synchronous)"""
    if _model is None:
        raise RuntimeError("Model not initialized")
    
    try:
        segments, info = _model.transcribe(
            audio_data,
            language=LANGUAGE,
            task=TASK,
            beam_size=BEAM_SIZE,
            best_of=BEST_OF,
            temperature=TEMPERATURE,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        # Process segments
        result_segments = []
        full_text = ""
        
        for segment in segments:
            segment_data = {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "words": []
            }
            
            if hasattr(segment, 'words') and segment.words:
                for word in segment.words:
                    segment_data["words"].append({
                        "start": word.start,
                        "end": word.end,
                        "word": word.word,
                        "probability": word.probability
                    })
            
            result_segments.append(segment_data)
            full_text += segment.text
        
        return {
            "text": full_text.strip(),
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "segments": result_segments
        }
        
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}")

class AudioBuffer:
    """Manages audio buffering and VAD detection"""
    
    def __init__(self):
        self.buffer = bytearray()
        self.is_speech = False
        self.silence_start = None
        self.last_speech_time = time.time()
        
    def add_audio(self, data: bytes) -> bool:
        """Add audio data and return True if we should process accumulated audio"""
        self.buffer.extend(data)
        
        # Check if we have enough data for VAD (30ms frames)
        frame_size = SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS * 30 // 1000  # 30ms
        
        if len(self.buffer) >= frame_size:
            # Extract frame for VAD
            frame_data = bytes(self.buffer[-frame_size:])
            
            try:
                contains_speech = _vad.is_speech(frame_data, SAMPLE_RATE)
                
                current_time = time.time()
                
                if contains_speech:
                    self.is_speech = True
                    self.last_speech_time = current_time
                    self.silence_start = None
                else:
                    if self.is_speech and self.silence_start is None:
                        self.silence_start = current_time
                    
                    # Check if silence threshold exceeded
                    if (self.silence_start and 
                        (current_time - self.silence_start) * 1000 >= SILENCE_THRESHOLD_MS):
                        return True
                        
            except Exception as e:
                print(f"[ASR] VAD error: {e}")
        
        # Check maximum audio length
        audio_length_ms = len(self.buffer) * 1000 // (SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS)
        if audio_length_ms >= MAX_AUDIO_LENGTH_MS:
            return True
            
        return False
    
    def get_audio_array(self) -> Optional[np.ndarray]:
        """Convert buffer to numpy array for Whisper"""
        if not self.buffer:
            return None
            
        # Check minimum length
        audio_length_ms = len(self.buffer) * 1000 // (SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS)
        if audio_length_ms < MIN_AUDIO_LENGTH_MS:
            return None
        
        # Convert to numpy array
        audio_data = np.frombuffer(self.buffer, dtype=np.int16)
        
        # Convert to float32 and normalize
        audio_float = audio_data.astype(np.float32) / 32768.0
        
        return audio_float
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()
        self.is_speech = False
        self.silence_start = None

async def handle_transcription_request(ws: ServerConnection, audio_buffer: AudioBuffer, request_id: str):
    """Process accumulated audio for transcription"""
    async with _sem:
        try:
            audio_data = audio_buffer.get_audio_array()
            if audio_data is None:
                await ws.send(json.dumps({
                    "type": "result",
                    "request_id": request_id,
                    "text": "",
                    "message": "Audio too short"
                }))
                return
            
            print(f"[ASR] Processing audio: {len(audio_data)} samples, {len(audio_data)/SAMPLE_RATE:.2f}s")
            
            # Send processing status
            await ws.send(json.dumps({
                "type": "processing",
                "request_id": request_id
            }))
            
            # Transcribe
            t0 = time.time()
            result = await asyncio.get_event_loop().run_in_executor(
                None, transcribe_audio_sync, audio_data
            )
            processing_time = time.time() - t0
            
            print(f"[ASR] Transcription completed in {processing_time:.2f}s: '{result['text'][:100]}...'")
            
            # Send result
            await ws.send(json.dumps({
                "type": "result",
                "request_id": request_id,
                "text": result["text"],
                "language": result["language"],
                "language_probability": result["language_probability"],
                "duration": result["duration"],
                "processing_time": processing_time,
                "segments": result["segments"]
            }))
            
        except Exception as e:
            print(f"[ASR] Transcription error: {e}")
            await ws.send(json.dumps({
                "type": "error",
                "request_id": request_id,
                "error": str(e)
            }))

async def handle_stream_mode(ws: ServerConnection):
    """Handle continuous audio streaming with real-time transcription"""
    audio_buffer = AudioBuffer()
    request_counter = 0
    
    try:
        await ws.send(json.dumps({
            "type": "ready",
            "sample_rate": SAMPLE_RATE,
            "channels": CHANNELS,
            "sample_width": SAMPLE_WIDTH,
            "format": "pcm_s16le"
        }))
        
        async for message in ws:
            if isinstance(message, bytes):
                # Audio data received
                should_process = audio_buffer.add_audio(message)
                
                if should_process:
                    request_counter += 1
                    request_id = f"stream-{request_counter}"
                    
                    # Process in background
                    asyncio.create_task(
                        handle_transcription_request(ws, audio_buffer, request_id)
                    )
                    
                    # Clear buffer for next chunk
                    audio_buffer.clear()
            else:
                # Text message (JSON)
                try:
                    data = json.loads(message)
                    msg_type = data.get("type")
                    
                    if msg_type == "start_stream":
                        audio_buffer.clear()
                        await ws.send(json.dumps({"type": "stream_started"}))
                    elif msg_type == "stop_stream":
                        # Process any remaining audio
                        if len(audio_buffer.buffer) > 0:
                            request_counter += 1
                            request_id = f"final-{request_counter}"
                            await handle_transcription_request(ws, audio_buffer, request_id)
                        audio_buffer.clear()
                        await ws.send(json.dumps({"type": "stream_stopped"}))
                        
                except json.JSONDecodeError:
                    await ws.send(json.dumps({
                        "type": "error",
                        "error": "Invalid JSON message"
                    }))
                    
    except (ConnectionClosedOK, ConnectionClosedError, ConnectionClosed):
        pass
    except Exception as e:
        print(f"[ASR] Stream error: {e}")

async def handle_single_request(ws: ServerConnection, request: Dict[str, Any]):
    """Handle single transcription request"""
    request_id = request.get("request_id", f"req-{uuid.uuid4()}")
    
    try:
        await ws.send(json.dumps({
            "type": "ready",
            "request_id": request_id,
            "sample_rate": SAMPLE_RATE,
            "channels": CHANNELS,
            "sample_width": SAMPLE_WIDTH,
            "format": "pcm_s16le"
        }))
        
        audio_buffer = AudioBuffer()
        timeout_task = None
        
        try:
            # Set up timeout
            if REQ_TIMEOUT_S > 0:
                timeout_task = asyncio.create_task(asyncio.sleep(REQ_TIMEOUT_S))
            
            # Collect audio data
            async for message in ws:
                if isinstance(message, bytes):
                    audio_buffer.add_audio(message)
                else:
                    # Check for end signal
                    try:
                        data = json.loads(message)
                        if data.get("type") == "audio_end":
                            break
                    except json.JSONDecodeError:
                        pass
                
                # Check timeout
                if timeout_task and timeout_task.done():
                    raise asyncio.TimeoutError("Request timeout")
            
            # Process the audio
            await handle_transcription_request(ws, audio_buffer, request_id)
            
        finally:
            if timeout_task and not timeout_task.done():
                timeout_task.cancel()
                
    except asyncio.TimeoutError:
        await ws.send(json.dumps({
            "type": "error",
            "request_id": request_id,
            "error": "Request timeout"
        }))
    except Exception as e:
        await ws.send(json.dumps({
            "type": "error",
            "request_id": request_id,
            "error": str(e)
        }))

async def handle_client(ws: ServerConnection):
    """Main client handler"""
    print("[ASR] Client connected")
    
    try:
        # Wait for initial message to determine mode
        first_message = await asyncio.wait_for(ws.recv(), timeout=10)
        
        if isinstance(first_message, bytes):
            # Binary first message means streaming mode
            await handle_stream_mode(ws)
        else:
            # JSON first message
            try:
                data = json.loads(first_message)
                msg_type = data.get("type")
                
                if msg_type == "transcribe":
                    await handle_single_request(ws, data)
                elif msg_type == "stream":
                    await handle_stream_mode(ws)
                else:
                    await ws.send(json.dumps({
                        "type": "error",
                        "error": f"Unknown message type: {msg_type}"
                    }))
            except json.JSONDecodeError:
                await ws.send(json.dumps({
                    "type": "error",
                    "error": "Invalid JSON in first message"
                }))
                
    except asyncio.TimeoutError:
        await ws.send(json.dumps({
            "type": "error",
            "error": "No initial message received"
        }))
    except (ConnectionClosedOK, ConnectionClosedError, ConnectionClosed):
        pass
    except Exception as e:
        print(f"[ASR] Client error: {e}")
    finally:
        print("[ASR] Client disconnected")

async def main():
    """Main entry point"""
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8762"))
    
    print("[ASR] Initializing Whisper model...")
    init_model_once()
    
    print("[ASR] Running model warmup...")
    await warmup_model()
    
    print(f"[ASR] Server ready on ws://{host}:{port}")
    print(f"[ASR] Model: {MODEL_SIZE}, Device: {DEVICE}, Audio: {SAMPLE_RATE}Hz {CHANNELS}ch")
    print(f"[ASR] VAD mode: {VAD_MODE}, Silence threshold: {SILENCE_THRESHOLD_MS}ms")
    
    async with serve(
        handle_client, host, port,
        max_size=None, max_queue=16,
        ping_interval=20, ping_timeout=20
    ):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())