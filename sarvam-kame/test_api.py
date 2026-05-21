import asyncio
import struct
from server.sarvam_client import transcribe, get_oracle, synthesize

async def test():
    # Test TTS
    audio, lat = await synthesize('Hello, this is a test.')
    print(f'TTS: {len(audio)} bytes, {lat:.2f}s')

    # Test LLM
    resp, lat = await get_oracle('Hello', [])
    print(f'LLM: {resp[:50]}..., {lat:.2f}s')

    # Test STT - create 1 second of silent audio
    pcm = struct.pack('<' + 'h'*16000, *([0]*16000))
    text, lat = await transcribe(pcm)
    print(f'STT: "{text}", {lat:.2f}s')

asyncio.run(test())