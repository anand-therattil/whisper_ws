#!/usr/bin/env python3
# whisper_client.py
# Simple client to test the Whisper WebSocket server

import asyncio
import json
import sys
import os
import wave
import argparse
from pathlib import Path
import websockets
import numpy as np
from scipy.io import wavfile
import librosa

class WhisperClient:
    def __init__(self, server_url="ws://localhost:8765"):
        self.server_url = server_url
        
    async def transcribe_file(self, audio_file_path):
        """Transcribe an audio file"""
        print(f"Loading audio file: {audio_file_path}")
        
        # Load and convert audio to the required format
        audio_data, sample_rate = self._load_audio(audio_file_path)
        
        print(f"Connecting to {self.server_url}...")
        
        try:
            async with websockets.connect(self.server_url) as websocket:
                # Send initial request
                request = {
                    "type": "transcribe",
                    "request_id": "test-001"
                }
                await websocket.send(json.dumps(request))
                
                # Wait for ready response
                response = await websocket.recv()
                ready_msg = json.loads(response)
                
                if ready_msg.get("type") != "ready":
                    print(f"Unexpected response: {ready_msg}")
                    return None
                
                print("Server ready, sending audio data...")
                
                # Send audio in chunks
                chunk_size = 4096  # bytes
                total_sent = 0
                
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:i + chunk_size]
                    await websocket.send(chunk)
                    total_sent += len(chunk)
                    
                    if total_sent % (chunk_size * 10) == 0:  # Progress every ~40KB
                        print(f"Sent {total_sent / 1024:.1f} KB...")
                
                print(f"Audio sent ({total_sent / 1024:.1f} KB total)")
                
                # Send end signal
                await websocket.send(json.dumps({"type": "audio_end"}))
                print("Waiting for transcription...")
                
                # Receive responses
                while True:
                    response = await websocket.recv()
                    data = json.loads(response)
                    
                    msg_type = data.get("type")
                    
                    if msg_type == "processing":
                        print("Server is processing audio...")
                    elif msg_type == "result":
                        return data
                    elif msg_type == "error":
                        print(f"Error: {data.get('error')}")
                        return None
                    else:
                        print(f"Unknown message type: {msg_type}")
                        
        except Exception as e:
            print(f"Connection error: {e}")
            return None
    
    def _load_audio(self, file_path):
        """Load audio file and convert to required format (16kHz, mono, s16le)"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        # Load audio using librosa (handles many formats)
        try:
            # Load and resample to 16kHz, convert to mono
            audio, sr = librosa.load(file_path, sr=16000, mono=True)
            
            # Convert to int16 format
            audio_int16 = (audio * 32767).astype(np.int16)
            
            # Convert to bytes
            audio_bytes = audio_int16.tobytes()
            
            print(f"Audio loaded: {len(audio):.2f}s, {sr}Hz, {len(audio_bytes)} bytes")
            return audio_bytes, sr
            
        except Exception as e:
            # Fallback: try with wave module for WAV files
            if file_path.suffix.lower() == '.wav':
                return self._load_wav_file(file_path)
            else:
                raise RuntimeError(f"Could not load audio file: {e}")
    
    def _load_wav_file(self, file_path):
        """Load WAV file using wave module"""
        with wave.open(str(file_path), 'rb') as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            
            print(f"WAV info: {sample_rate}Hz, {channels}ch, {sample_width*8}bit")
            
            # Convert to numpy array
            if sample_width == 1:
                audio_data = np.frombuffer(frames, dtype=np.uint8)
                audio_data = ((audio_data.astype(np.float32) - 128) / 128.0)
            elif sample_width == 2:
                audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            elif sample_width == 4:
                audio_data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")
            
            # Convert to mono if stereo
            if channels == 2:
                audio_data = audio_data.reshape(-1, 2).mean(axis=1)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            
            # Convert back to int16 bytes
            audio_int16 = (audio_data * 32767).astype(np.int16)
            return audio_int16.tobytes(), 16000

def print_result(result):
    """Pretty print the transcription result"""
    if not result:
        print("No result received")
        return
    
    print("\n" + "="*60)
    print("TRANSCRIPTION RESULT")
    print("="*60)
    
    text = result.get("text", "")
    print(f"Text: {text}")
    print()
    
    language = result.get("language")
    lang_prob = result.get("language_probability")
    if language:
        print(f"Language: {language} (confidence: {lang_prob:.2f})")
    
    duration = result.get("duration")
    processing_time = result.get("processing_time")
    if duration:
        print(f"Audio duration: {duration:.2f}s")
    if processing_time:
        print(f"Processing time: {processing_time:.2f}s")
        if duration:
            print(f"Real-time factor: {processing_time/duration:.2f}x")
    
    segments = result.get("segments", [])
    if segments:
        print(f"\nSegments ({len(segments)}):")
        print("-" * 40)
        for i, segment in enumerate(segments, 1):
            start = segment.get("start", 0)
            end = segment.get("end", 0)
            seg_text = segment.get("text", "")
            print(f"{i:2d}. [{start:6.2f}s - {end:6.2f}s] {seg_text}")
    
    print("="*60)

async def main():
    parser = argparse.ArgumentParser(description="Test Whisper WebSocket Server")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--server", default="ws://localhost:8762", 
                       help="WebSocket server URL (default: ws://localhost:8765)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}")
        sys.exit(1)
    
    client = WhisperClient(args.server)
    
    print("Whisper WebSocket Client")
    print(f"Audio file: {args.audio_file}")
    print(f"Server: {args.server}")
    print()
    
    try:
        result = await client.transcribe_file(args.audio_file)
        print_result(result)
        
        # Also save result to JSON file
        if result:
            output_file = Path(args.audio_file).stem + "_transcription.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\nResult saved to: {output_file}")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Check dependencies
    try:
        import websockets
        import librosa
        import scipy
    except ImportError as e:
        print("Missing dependencies. Install with:")
        print("pip install websockets librosa scipy")
        sys.exit(1)
    
    asyncio.run(main())