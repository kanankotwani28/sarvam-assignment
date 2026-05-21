"""
Tandem engine: pre-synthesizes filler phrases at startup,
returns them instantly during Mode B while oracle runs in background.
"""

import asyncio
import time
import random


FILLER_PHRASES = {
    "en-IN": [
        "Let me look that up for you.",
        "One moment please.",
        "Give me a second.",
        "I'm checking on that.",
        "Just a moment.",
    ],
    "hi-IN": [
        "कृपया एक क्षण प्रतीक्षा करें।",
        "मैं देख रहा हूँ।",
        "एक पल रुकिए।",
        "मैं अभी जवाब लेकर आता हूँ।",
        "ज़रा रुकिए।",
    ],
}

DEFAULT_LANG = "en-IN"


class TandemEngine:
    """Returns pre-cached filler audio in <1ms while oracle computes the real answer."""

    def __init__(self):
        self.fillers: dict[str, list[bytes]] = {}
        self.loaded = False
        self.preload_latencies: dict[str, float] = {}
        self._turn_count = 0
        self._filler_sent_count = 0
        self._total_filler_latency = 0.0
        self._total_oracle_latency = 0.0
        self._turn_latencies: list[dict] = []

    async def preload_fillers(self):
        """
        Synthesize all filler phrases via Sarvam TTS at startup.
        Called once from the server startup event.
        """
        from .sarvam_client import synthesize

        print("[TANDEM] Preloading filler phrases...")
        for lang, phrases in FILLER_PHRASES.items():
            self.fillers[lang] = []
            total_lat = 0.0
            for phrase in phrases:
                audio, lat = await synthesize(phrase, language=lang)
                if audio:
                    self.fillers[lang].append(audio)
                    total_lat += lat
                    print(f"[TANDEM]  {lang}: phrase -> {len(audio)} bytes ({lat:.2f}s)")
                else:
                    print(f"[TANDEM]  {lang}: phrase - TTS failed, skipping")
            avg = total_lat / max(len(phrases), 1)
            self.preload_latencies[lang] = avg
            print(f"[TANDEM]  {lang}: avg TTS {avg:.3f}s across {len(self.fillers[lang])} fillers")

        if any(self.fillers.values()):
            self.loaded = True
            print("[TANDEM] Filler preloading complete!")
        else:
            print("[TANDEM] WARNING: No fillers loaded — Mode B will have no fast path")
            # Generate a minimal silence filler as fallback
            silence = b'\x00\x00' * 24000  # 1s of silence at 24kHz 16-bit
            self.fillers[DEFAULT_LANG] = [silence]
            self.loaded = True

    def get_filler(self, language: str = DEFAULT_LANG) -> bytes:
        """
        Return the next filler audio in round-robin order.
        Takes <1ms — just reads from memory.
        """
        lang = language if language in self.fillers else DEFAULT_LANG
        pool = self.fillers.get(lang, self.fillers.get(DEFAULT_LANG, []))
        if not pool:
            return b""
        idx = self._turn_count % len(pool)
        self._turn_count += 1
        return pool[idx]

    def get_filler_stats(self) -> dict:
        """Return filler preload stats for the /stats endpoint."""
        return {
            "fillers_loaded": sum(len(v) for v in self.fillers.values()),
            "languages": list(self.fillers.keys()),
            "preload_avg_tts_seconds": self.preload_latencies,
        }


# ── Global singleton ──
tandem = TandemEngine()
