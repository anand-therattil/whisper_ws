# ws_client_python.py
# Usage:
#   pip install websockets
#   python ws_client_python.py --file sample_16k_mono.wav --url ws://localhost:8762
import argparse
import asyncio
import json
import wave
import math
from websockets import connect

CHUNK_MS = 20  # send 20ms chunks
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
CHANNELS = 1

def wav_iter_frames(path, chunk_ms=CHUNK_MS):
    """Yield PCM16LE bytes of chunk_ms from a 16k mono 16-bit wav file.
    Expects input already in the required format. No resampling done here."""
    with wave.open(path, "rb") as wf:
        assert wf.getsampwidth() == SAMPLE_WIDTH, "expected 16-bit wav"
        assert wf.getnchannels() == CHANNELS, "expected mono wav"
        assert wf.getframerate() == SAMPLE_RATE, f"expected {SAMPLE_RATE}Hz wav"
        frames_per_chunk = int(SAMPLE_RATE * (chunk_ms / 1000.0))
        while True:
            data = wf.readframes(frames_per_chunk)
            if not data:
                break
            yield data

async def run_file_stream(ws_url, wav_path, auth_token=None):
    headers = []
    # for websockets client we can pass headers as list of tuples
    if auth_token:
        headers.append(("Authorization", f"Bearer {auth_token}"))

    async with connect(ws_url, extra_headers=headers, max_size=None) as ws:
        # announce stream mode
        await ws.send(json.dumps({"type": "stream"}))
        # server should respond with ready
        print("Sent stream start, waiting for server messages in background...")

        async def recv_loop():
            try:
                async for msg in ws:
                    try:
                        obj = json.loads(msg)
                        print("[SERVER]", obj)
                    except Exception:
                        print("[SERVER non-json]", msg)
            except Exception as e:
                print("recv_loop ended:", e)

        recv_task = asyncio.create_task(recv_loop())

        # send audio chunks
        for chunk in wav_iter_frames(wav_path):
            await ws.send(chunk)  # binary frame
            # pace to real-time (optional). If you want faster-than-realtime, remove sleep.
            await asyncio.sleep(CHUNK_MS / 1000.0)

        # send stop_stream control so server will flush & transcribe remaining
        await ws.send(json.dumps({"type": "stop_stream"}))
        # wait a bit for server result(s)
        await asyncio.sleep(2.0)
        recv_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await recv_task

if __name__ == "__main__":
    import contextlib
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://localhost:8762")
    parser.add_argument("--file", required=True, help="16kHz mono 16-bit wav file")
    parser.add_argument("--auth", default=None, help="optional bearer token")
    args = parser.parse_args()
    asyncio.run(run_file_stream(args.url, args.file, auth_token=args.auth))
