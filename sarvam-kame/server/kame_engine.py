"""
Tandem engine: pre-synthesizes filler phrases at startup,
returns them instantly during Mode B while oracle runs in background.
Supports question-type classification for context-aware fillers.
"""

import asyncio
import time
import re


# ── Filler phrases by category ──────────────────────────
FILLER_CATEGORIES = {
    "general": {
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
    },
    "knowledge": {
        "en-IN": [
            "Let me find that information.",
            "Looking that up now.",
            "Let me search for that.",
            "Give me a moment to find out.",
        ],
        "hi-IN": [
            "मैं वह जानकारी ढूंढ रहा हूँ।",
            "मैं अभी देखता हूँ।",
            "वह जानकारी लेकर आता हूँ।",
            "एक क्षण, मैं देख रहा हूँ।",
        ],
    },
    "time_date": {
        "en-IN": [
            "Let me check the time.",
            "Checking the time now.",
            "One moment, checking the date and time.",
        ],
        "hi-IN": [
            "मैं समय देख रहा हूँ।",
            "समय देख रहा हूँ।",
            "एक पल, समय और तारीख देख रहा हूँ।",
        ],
    },
    "math": {
        "en-IN": [
            "Let me calculate that.",
            "Working out the math.",
            "Doing the calculation now.",
        ],
        "hi-IN": [
            "मैं हिसाब कर रहा हूँ।",
            "गणना कर रहा हूँ।",
            "हिसाब लगा रहा हूँ।",
        ],
    },
}

DEFAULT_CATEGORY = "general"
DEFAULT_LANG = "en-IN"

# ── Question-type classifier (keyword-based) ────────────
_CATEGORY_PATTERNS = [
    ("time_date",    [r'\btime\b', r'\bclock\b', r'\bdate\b', r'\bday\b',
                      r'\bhour\b', r'\bminute\b',
                      r'\bकितने बजे\b', r'\bसमय\b', r'\bतारीख\b']),
    ("math",         [r'\bcalculate\b', r'\bcomput', r'\bsum\b',
                      r'\bplus\b', r'\bminus\b', r'\btimes\b',
                      r'\bdivid', r'\bmath\b', r'\bcount\b',
                      r'\bहिसाब\b', r'\bगणना\b', r'\bजोड़\b']),
    ("knowledge",    [r'\bwho\b', r'\bwhat\b', r'\bwhy\b', r'\bhow\b',
                      r'\bwhere\b', r'\bwhen\b', r'\bwhich\b',
                      r'\bdefine\b', r'\bmeaning\b', r'\btell me\b',
                      r'\bexplain\b',
                      r'\bकौन\b', r'\bक्या\b', r'\bक्यों\b', r'\bकैसे\b',
                      r'\bकहाँ\b', r'\bबताओ\b', r'\bसमझाओ\b']),
]


def classify_question(text: str) -> str:
    """Classify utterance text into a filler category."""
    if not text:
        return DEFAULT_CATEGORY
    text_lower = text.lower()
    for category, patterns in _CATEGORY_PATTERNS:
        for pat in patterns:
            if re.search(pat, text_lower):
                return category
    return DEFAULT_CATEGORY


class TandemEngine:
    """Returns pre-cached filler audio in <1ms while oracle computes the real answer."""

    def __init__(self):
        self.fillers: dict[str, dict[str, list[bytes]]] = {}  # lang -> category -> [audio]
        self.loaded = False
        self.preload_latencies: dict[str, float] = {}
        self._turn_count = 0
        self._filler_sent_count = 0
        self._total_filler_latency = 0.0
        self._total_oracle_latency = 0.0
        self._turn_latencies: list[dict] = []

    async def preload_minimal(self):
        """
        Quick preload of 1 filler per language (~5s total).
        Called at startup before accepting connections — ensures filler is always ready.
        """
        from .sarvam_client import synthesize

        print("[TANDEM] Preloading minimal fillers...")
        for lang in ("en-IN", "hi-IN"):
            self.fillers[lang] = {}
            phrase = FILLER_CATEGORIES["general"][lang][0]  # first phrase only
            audio, lat = await synthesize(phrase, language=lang)
            if audio:
                self.fillers[lang] = {"general": [audio]}
                print(f"[TANDEM]  {lang}/general: {len(audio)} bytes ({lat:.2f}s)")
            else:
                self.fillers[lang] = {"general": [b'\x00\x00' * 24000]}
                print(f"[TANDEM]  {lang}/general: TTS failed, silence fallback")

        self.loaded = True
        print("[TANDEM] Minimal preload done — fillers ready.")

    async def preload_fillers(self):
        """
        Full preload of all categories + remaining phrases.
        Runs in background after minimal preload is done.
        """
        from .sarvam_client import synthesize

        print("[TANDEM] Full preload (background)...")
        all_langs = ["en-IN", "hi-IN"]
        for lang in all_langs:
            self.fillers.setdefault(lang, {}).setdefault("general", [])
            for category, phrases in FILLER_CATEGORIES.items():
                if lang not in phrases:
                    continue
                self.fillers[lang].setdefault(category, [])
                for phrase in phrases[lang]:
                    # Skip already-loaded phrases
                    if category == "general" and self.fillers[lang]["general"]:
                        continue
                    audio, lat = await synthesize(phrase, language=lang)
                    if audio:
                        self.fillers[lang][category].append(audio)
                        print(f"[TANDEM]  {lang}/{category}: {len(audio)} bytes ({lat:.2f}s)")

        print("[TANDEM] Full preload complete!")

    def get_filler(self, language: str = DEFAULT_LANG,
                   category: str = DEFAULT_CATEGORY) -> bytes:
        """
        Return the next filler audio in round-robin order.
        Takes <1ms — just reads from memory.
        """
        lang = language if language in self.fillers else DEFAULT_LANG
        pool = self.fillers.get(lang, {}).get(category,
                self.fillers.get(lang, {}).get(DEFAULT_CATEGORY,
                    self.fillers.get(DEFAULT_LANG, {}).get(DEFAULT_CATEGORY, [])))
        if not pool:
            return b""
        idx = self._turn_count % len(pool)
        self._turn_count += 1
        return pool[idx]

    def get_filler_stats(self) -> dict:
        """Return filler preload stats for the /stats endpoint."""
        lang_details = {}
        for lang, cats in self.fillers.items():
            lang_details[lang] = {cat: len(audios) for cat, audios in cats.items()}
        return {
            "fillers_loaded": sum(sum(v.values()) for v in self.fillers.values()),
            "categories": list(FILLER_CATEGORIES.keys()),
            "languages": list(self.fillers.keys()),
            "fillers_by_lang": lang_details,
        }


# ── Global singleton ──
tandem = TandemEngine()
