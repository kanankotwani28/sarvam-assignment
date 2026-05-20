import asyncio
import re

from server.sarvam_client import get_oracle, synthesize

async def test():
    text, lat = await get_oracle("Hello, who are you?", [])

    # Remove leaked reasoning
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    print(f"LLM: '{text}' in {lat:.2f}s")

    audio, lat2 = await synthesize(text)

    print(f"TTS: {len(audio)} bytes in {lat2:.2f}s")

    with open("test_output.wav", "wb") as f:
        f.write(audio)

    print("Saved test_output.wav")

asyncio.run(test())