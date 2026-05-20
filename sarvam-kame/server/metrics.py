import time
import json
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

LOG_FILE = Path("latency_log.jsonl")

@dataclass
class SessionMetrics:
    session_id:          str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    mode:                str   = "cascaded"
    # timestamps
    t_speech_start:      Optional[float] = None
    t_speech_end:        Optional[float] = None
    t_stt_done:          Optional[float] = None
    t_llm_done:          Optional[float] = None
    t_tts_done:          Optional[float] = None
    t_first_audio_sent:  Optional[float] = None
    # computed
    transcript:          str   = ""
    response_text:       str   = ""

    # ── derived latencies ──────────────────────
    @property
    def stt_latency(self) -> Optional[float]:
        if self.t_stt_done and self.t_speech_end:
            return round(self.t_stt_done - self.t_speech_end, 3)

    @property
    def llm_latency(self) -> Optional[float]:
        if self.t_llm_done and self.t_stt_done:
            return round(self.t_llm_done - self.t_stt_done, 3)

    @property
    def tts_latency(self) -> Optional[float]:
        if self.t_tts_done and self.t_llm_done:
            return round(self.t_tts_done - self.t_llm_done, 3)

    @property
    def total_latency(self) -> Optional[float]:
        if self.t_first_audio_sent and self.t_speech_end:
            return round(self.t_first_audio_sent - self.t_speech_end, 3)

    @property
    def time_to_first_audio(self) -> Optional[float]:
        """For KAME: how soon after user starts speaking do they hear something."""
        if self.t_first_audio_sent and self.t_speech_start:
            return round(self.t_first_audio_sent - self.t_speech_start, 3)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update({
            "stt_latency":        self.stt_latency,
            "llm_latency":        self.llm_latency,
            "tts_latency":        self.tts_latency,
            "total_latency":      self.total_latency,
            "time_to_first_audio":self.time_to_first_audio,
        })
        return d

    def log(self):
        d = self.to_dict()
        print(f"\n{'─'*55}")
        print(f"  session  : {self.session_id}  [{self.mode}]")
        print(f"  transcript : {self.transcript[:60]}")
        print(f"  response   : {self.response_text[:60]}")
        print(f"  STT        : {self.stt_latency}s")
        print(f"  LLM        : {self.llm_latency}s")
        print(f"  TTS        : {self.tts_latency}s")
        print(f"  total      : {self.total_latency}s")
        print(f"  1st audio  : {self.time_to_first_audio}s")
        print(f"{'─'*55}\n")

        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(d) + "\n")