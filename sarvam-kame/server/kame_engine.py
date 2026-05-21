"""
Tandem engine: pre-synthesizes filler phrases at startup,
returns them instantly during Mode B while oracle runs in background.
Supports question-type classification for context-aware fillers.
"""

import asyncio
import time
import re
import struct
import io


# ── Filler phrases by category ──────────────────────────
# Each phrase is 2-4 sentences — natural, conversational, ~4-8s spoken.
# Only 2 per category keeps startup preload fast.
FILLER_CATEGORIES = {
    "general": {
        "en-IN": [
            "Let me look that up for you. I'll find the information you need and be right back with an answer. Give me just a moment.",
            "Thanks for your question. Let me gather the details for you. I'll have everything ready in just a few seconds.",
            "Sure, let me check on that. I'm pulling together the information now and will have an answer for you shortly.",
        ],
        "hi-IN": [
            "आपके सवाल के लिए शुक्रिया। मैं अभी आपके लिए जानकारी ढूंढ रहा हूँ। कृपया एक पल रुकिए, मैं तुरंत जवाब लेकर आ रहा हूँ।",
            "मैं देख रहा हूँ। सारी जानकारी इकट्ठा कर रहा हूँ। बस कुछ ही पलों में आपको जवाब मिल जाएगा।",
        ],
    },
    "knowledge": {
        "en-IN": [
            "That's a great question. Let me search through my knowledge base to find the most accurate information for you. I'll have an answer in just a moment.",
            "Let me find that information for you. I'm looking through my sources to make sure I give you the best possible answer. Bear with me for a few seconds.",
            "I'm looking that up right now. There's quite a bit of information on this topic, so let me find the most relevant details for you. I'll be right back.",
        ],
        "hi-IN": [
            "यह बहुत अच्छा सवाल है। मैं आपके लिए सबसे सटीक जानकारी ढूंढ रहा हूँ। बस एक पल का धैर्य रखें, मैं अभी जवाब लेकर आता हूँ।",
            "मैं वह जानकारी खोज रहा हूँ। कई स्रोतों से सही जानकारी निकाल रहा हूँ ताकि आपको सबसे अच्छा जवाब मिल सके।",
        ],
    },
    "time_date": {
        "en-IN": [
            "Let me check the current time and date for you. I'm looking up the accurate details right now and will tell you in just a moment.",
            "I'll check the time for you right away. Let me look up the exact current time and date from my sources.",
        ],
        "hi-IN": [
            "मैं अभी आपके लिए समय और तारीख देख रहा हूँ। सटीक जानकारी लेकर एक पल में आपको बताता हूँ।",
            "समय देख रहा हूँ। वर्तमान समय और तारीख की पुष्टि कर रहा हूँ, बस एक सेकंड।",
        ],
    },
    "math": {
        "en-IN": [
            "Let me work out the calculation for you. I'm going through the numbers step by step to make sure everything is accurate. I'll have the result in just a moment.",
            "I'm calculating that right now. Let me double check the numbers to ensure the answer is correct. Give me just a few seconds and I'll have it for you.",
        ],
        "hi-IN": [
            "मैं आपके लिए हिसाब कर रहा हूँ। सभी संख्याओं को ध्यान से जाँच रहा हूँ ताकि सही जवाब मिल सके। बस एक पल में परिणाम दे दूँगा।",
            "गणना कर रहा हूँ। एक बार सब कुछ दोबारा जाँच लेता हूँ ताकि कोई गलती न रहे। तुरंत जवाब दे रहा हूँ।",
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

    @staticmethod
    def _concat_wavs(wavs: list[bytes]) -> bytes:
        """Concatenate multiple WAV files into one continuous WAV."""
        if not wavs:
            return b""
        if len(wavs) == 1:
            return wavs[0]
        # Parse first WAV header (44 bytes, 16-bit mono)
        header = wavs[0][:44]
        data_size = struct.unpack_from('<I', header, 40)[0]
        chunks = [wavs[0]]  # first WAV with header
        for w in wavs[1:]:
            # Strip 44-byte WAV header from subsequent WAVs, keep raw PCM
            if len(w) > 44:
                chunks.append(w[44:])
                data_size += len(w) - 44
        # Update data size in first header
        new_header = bytearray(header)
        struct.pack_into('<I', new_header, 4, 36 + data_size)  # RIFF size
        struct.pack_into('<I', new_header, 40, data_size)       # data chunk size
        chunks[0] = bytes(new_header)
        return b"".join(chunks)

    def get_filler(self, language: str = DEFAULT_LANG,
                   category: str = DEFAULT_CATEGORY,
                   count: int = 2) -> bytes:
        """
        Return a concatenated sequence of `count` filler phrases.
        Single WAV — plays continuously to cover oracle latency.
        Takes <1ms — just reads from memory.
        """
        lang = language if language in self.fillers else DEFAULT_LANG
        pool = self.fillers.get(lang, {}).get(category,
                self.fillers.get(lang, {}).get(DEFAULT_CATEGORY,
                    self.fillers.get(DEFAULT_LANG, {}).get(DEFAULT_CATEGORY, [])))
        if not pool:
            return b""
        wavs = []
        for i in range(count):
            idx = (self._turn_count + i) % len(pool)
            wavs.append(pool[idx])
        self._turn_count += count
        return self._concat_wavs(wavs)

    def get_bilingual_filler(self, category: str = DEFAULT_CATEGORY) -> bytes:
        """
        Return a WAV with one EN + one HI filler phrase concatenated.
        Used for the first turn when the user's language is unknown.
        """
        wavs = []
        for lang in ("en-IN", "hi-IN"):
            pool = self.fillers.get(lang, {}).get(category,
                    self.fillers.get(lang, {}).get(DEFAULT_CATEGORY, []))
            if pool:
                wavs.append(pool[self._turn_count % len(pool)])
        self._turn_count += 1
        return self._concat_wavs(wavs) if wavs else b""

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
