import os
from dotenv import load_dotenv

load_dotenv()

# Sarvam API — verified from docs.sarvam.ai
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
if not SARVAM_API_KEY:
    raise ValueError("SARVAM_API_KEY not set in .env")

SARVAM_STT_URL    = "https://api.sarvam.ai/speech-to-text"
SARVAM_TTS_URL    = "https://api.sarvam.ai/text-to-speech"
SARVAM_LLM_URL    = "https://api.sarvam.ai/v1/chat/completions"

# Audio settings
STT_SAMPLE_RATE   = 16000   # Sarvam STT requires 16kHz PCM
TTS_SAMPLE_RATE   = 24000   # Bulbul v3 default output

# Model settings
STT_MODEL         = "saaras:v3"          # best, supports codemix
LLM_MODEL         = "sarvam-m"           # use sarvam-30b if you have access
TTS_MODEL         = "bulbul:v3"
TTS_SPEAKER       = "simran"              # female Indian English voice

# Server
HOST              = "0.0.0.0"
PORT              = 8000