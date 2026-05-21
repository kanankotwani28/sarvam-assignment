import httpx
import base64
import struct
import time
import io
import re
from .config import (
    SARVAM_API_KEY, SARVAM_STT_URL, SARVAM_TTS_URL, SARVAM_LLM_URL,
    STT_MODEL, LLM_MODEL, TTS_MODEL, TTS_SPEAKER
)

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 16000,
               channels: int = 1, bits: int = 16) -> bytes:
    """Wrap raw PCM bytes in a WAV header — required by Sarvam STT."""
    data_size = len(pcm_bytes)
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + data_size, b'WAVE',
        b'fmt ', 16, 1, channels,
        sample_rate,
        sample_rate * channels * bits // 8,
        channels * bits // 8, bits,
        b'data', data_size
    )
    return header + pcm_bytes


def _sarvam_headers_key() -> dict:
    """Header for STT and TTS (api-subscription-key)."""
    return {"api-subscription-key": SARVAM_API_KEY}


def _sarvam_headers_bearer() -> dict:
    """Header for LLM (Bearer token — also works with api-subscription-key)."""
    return {
        "Authorization": f"Bearer {SARVAM_API_KEY}",
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json",
    }


# ─────────────────────────────────────────────
#  STT  —  POST https://api.sarvam.ai/speech-to-text
# ─────────────────────────────────────────────

async def transcribe(pcm_bytes: bytes, language: str = "unknown") -> tuple[str, float]:
    """
    Transcribe raw 16kHz mono PCM audio.
    Returns (transcript, latency_seconds).
    language: 'unknown' = auto-detect, 'hi-IN', 'en-IN', etc.
    """
    if len(pcm_bytes) < 3200:   # less than 0.1s — skip
        return "", 0.0

    wav_bytes = pcm_to_wav(pcm_bytes)
    t = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            resp = await client.post(
                SARVAM_STT_URL,
                headers=_sarvam_headers_key(),
                files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                data={
                    "model": STT_MODEL,
                    "language_code": language,
                    "mode": "codemix",  # handles Hindi+English mixed speech
                }
            )
        latency = time.perf_counter() - t

        if resp.status_code == 200:
            transcript = resp.json().get("transcript", "").strip()
            lang_detected = resp.json().get("language_code", "")
            print(f"[STT] '{transcript}' | lang={lang_detected} | {latency:.2f}s")
            return transcript, latency
        else:
            print(f"[STT] Error {resp.status_code}: {resp.text[:200]}")
            return "", latency

    except Exception as e:
        print(f"[STT] Exception: {e}")
        return "", time.perf_counter() - t


# ─────────────────────────────────────────────
#  LLM  —  POST https://api.sarvam.ai/v1/chat/completions
# ─────────────────────────────────────────────

def _clean_llm_response(text: str) -> str:
    """
    Remove all thinking/reasoning tags from LLM response.
    Handles: <think>, <Thinking>, <thinking>, <reasoning>, <Reasoning>
    """
    # <think>...</think> and <Thinking>...</Thinking> — both cases
    think_open = '<[Tt]hink'
    think_close = '/[Tt]hink>'
    pattern = think_open + '.*?(?:' + think_close + '|$)'
    text = re.sub(pattern, '', text, flags=re.DOTALL)

    # <reasoning>...</reasoning> — both cases  
    text = re.sub(r'<[Rr]easoning>.*?(?:</[Rr]easoning>|$)', '', text, flags=re.DOTALL)

    # Remove leftover markdown that breaks TTS
    text = re.sub(r'[*_`#]', '', text)

    # Remove any other stray XML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    text = text.strip()

    if not text:
        text = 'I am ready to help.'

    return text

# async def get_oracle(
#     transcript: str,
#     history: list[dict],
#     max_tokens: int = 120
# ) -> tuple[str, float]:
#     """
#     Generate oracle response from Sarvam LLM.
#     Returns (response_text, latency_seconds).
#     Keeps response short — oracle is a guide, not the final answer.
#     """
#     system_prompt = (
#         "You are a friendly Indian voice assistant named Sarvam. "
#         "Give short, natural responses in 1-2 sentences. "
#         "Never show your thinking or reasoning process. "
#         "Only output your final response. "
#         "If the user speaks Hindi or code-mixed language, reply in the same language."
#     )
#     messages = (
#         [{"role": "system", "content": system_prompt}]
#         + history[-8:]   # last 4 turns for context
#         + [{"role": "user", "content": transcript}]
#     )

#     t = time.perf_counter()
#     try:
#         async with httpx.AsyncClient(timeout=12.0) as client:
#             resp = await client.post(
#                 SARVAM_LLM_URL,
#                 headers=_sarvam_headers_bearer(),
#                 json={
#                     "model": LLM_MODEL,
#                     "messages": messages,
#                     "max_tokens": max_tokens,
#                     "temperature": 0.7,
#                     "top_p": 0.9,
#                 }
#             )
#         latency = time.perf_counter() - t

#         if resp.status_code == 200:
#             raw_text = resp.json()["choices"][0]["message"]["content"].strip()
#             text = _clean_llm_response(raw_text)
#             print(f"[LLM] raw: '{raw_text[:100]}...'")
#             print(f"[LLM] clean: '{text[:80]}...' | {latency:.2f}s")
#             return text, latency
#         else:
#             print(f"[LLM] Error {resp.status_code}: {resp.text[:200]}")
#             return "", latency

#     except Exception as e:
#         print(f"[LLM] Exception: {e}")
#         return "", time.perf_counter() - t


async def get_oracle(
    transcript: str,
    history: list[dict],
    max_tokens: int = 200
) -> tuple[str, float]:
    """
    Generate oracle response from Sarvam LLM.
    Returns (response_text, latency_seconds).
    Strips all thinking/reasoning blocks before returning.
    """
    system_prompt = (
        "You are Sarvam, an Indian voice assistant. "
        "Answer every question directly and completely. "
        "For math, calculate and give the exact number immediately. "
        "For things you don't know, say exactly: I don't have that information. "
        "Never say 'let me help' without finishing the sentence. "
        "Keep answers under 3 spoken sentences. "
        "Never use bullet points or markdown — only plain speech."
    )

    messages = (
        [{"role": "system", "content": system_prompt}]
        + history[-8:]
        + [{"role": "user", "content": transcript}]
    )

    t = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                SARVAM_LLM_URL,
                headers=_sarvam_headers_bearer(),
                json={
                    "model": LLM_MODEL,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.3,
                    "top_p": 0.9,
                }
            )
        latency = time.perf_counter() - t

        if resp.status_code == 200:
            data = resp.json()

            if not data.get("choices"):
                print(f"[LLM] Empty choices: {data}")
                return "I could not generate a response.", latency

            raw_text = data["choices"][0]["message"]["content"]
            print(f"[LLM] raw:   '{raw_text[:120]}'")

            clean_text = _clean_llm_response(raw_text)

            if not clean_text or clean_text == "I am ready to help.":
                print("[LLM] Response was entirely thinking block — retrying with simpler prompt")
                # One retry with explicit instruction to skip thinking
                messages[-1]["content"] = transcript + " (answer directly, no thinking)"
                async with httpx.AsyncClient(timeout=15.0) as client2:
                    resp2 = await client2.post(
                        SARVAM_LLM_URL,
                        headers=_sarvam_headers_bearer(),
                        json={
                            "model": LLM_MODEL,
                            "messages": messages,
                            "max_tokens": max_tokens,
                            "temperature": 0.3,
                        }
                    )
                if resp2.status_code == 200:
                    raw2 = resp2.json()["choices"][0]["message"]["content"]
                    clean_text = _clean_llm_response(raw2)

            print(f"[LLM] clean: '{clean_text[:80]}' | {latency:.2f}s")
            return clean_text, latency

        elif resp.status_code == 429:
            print("[LLM] Rate limited — waiting 2s")
            await asyncio.sleep(2)
            return "I am a bit busy right now. Please try again.", latency

        else:
            print(f"[LLM] Error {resp.status_code}: {resp.text[:300]}")
            return "I had trouble connecting. Please try again.", latency

    except httpx.TimeoutException:
        print("[LLM] Timeout — 20s exceeded")
        return "That took too long. Please try again.", time.perf_counter() - t

    except Exception as e:
        print(f"[LLM] Exception: {e}")
        return "Something went wrong. Please try again.", time.perf_counter() - t


# ─────────────────────────────────────────────
#  TTS  —  POST https://api.sarvam.ai/text-to-speech
# ─────────────────────────────────────────────

async def synthesize(text: str, language: str = "en-IN") -> tuple[bytes, float]:
    """
    Convert text to WAV audio bytes using Bulbul v3.
    Returns (wav_bytes, latency_seconds).
    wav_bytes is base64-decoded WAV, ready to send to browser.
    """
    if not text.strip():
        return b"", 0.0

    t = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                SARVAM_TTS_URL,
                headers=_sarvam_headers_key(),
                json={
                    "text": text[:2500],  # bulbul:v3 max 2500 chars
                    "target_language_code": language,
                    "speaker": TTS_SPEAKER,
                    "model": TTS_MODEL,
                    "speech_sample_rate": 24000,
                    "pace": 1.0,
                }
            )
        latency = time.perf_counter() - t

        if resp.status_code == 200:
            audio_b64 = resp.json()["audios"][0]
            audio_bytes = base64.b64decode(audio_b64)
            print(f"[TTS] {len(audio_bytes)} bytes | {latency:.2f}s")
            return audio_bytes, latency
        else:
            print(f"[TTS] Error {resp.status_code}: {resp.text[:200]}")
            return b"", latency

    except Exception as e:
        print(f"[TTS] Exception: {e}")
        return b"", time.perf_counter() - t