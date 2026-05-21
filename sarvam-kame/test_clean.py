import asyncio
import httpx
import re

def _clean_llm_response(text):
    text = re.sub(r'<think>.*?
</think>

', '', text, flags=re.DOTALL).strip()
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL).strip()
    return text

async def test():
    headers = {
        'Authorization': 'Bearer sk_1wabj05h_GD6nKUoSYocwwIY2nNZAy33X',
        'api-subscription-key': 'sk_1wabj05h_GD6nKUoSYocwwIY2nNZAy33X'
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            'https://api.sarvam.ai/v1/chat/completions',
            headers=headers,
            json={
                'model': 'sarvam-m',
                'messages': [
                    {'role': 'system', 'content': 'You are a friendly Indian voice assistant. Give short, natural responses in 1-2 sentences. Never show your thinking or reasoning process. Only output your final response.'},
                    {'role': 'user', 'content': 'Hello, my name is Kanan.'}
                ],
                'max_tokens': 50,
                'temperature': 0.7,
                'top_p': 0.9
            }
        )
        raw = resp.json()['choices'][0]['message']['content']
        clean = _clean_llm_response(raw)
        print('RAW:', raw[:200])
        print('---')
        print('CLEAN:', clean)

asyncio.run(test())
