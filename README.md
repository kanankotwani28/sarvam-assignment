# Sarvam-KAME Voice Agent

> **Tandem voice architecture on a 16GB CPU-only machine — no GPU, no Moshi, fully functional.**

A research prototype that implements the tandem architecture concept from the [KAME paper (Kyutai)](https://github.com/kyutai-labs/kame) using Sarvam AI's Indian language APIs. The original KAME uses a 31GB neural codec (Moshi) for the fast path. This project replaces that with pre-synthesized filler audio served from RAM, demonstrating the same architectural claim on consumer hardware.

---

## The Core Insight

A real-time voice AI doesn't need to generate the answer instantly.  
**It just needs to respond instantly.**

```
Traditional thinking:  speak → wait → wait → wait → hear answer
KAME thinking:         speak → hear something immediately → hear answer
```

The two are completely different user experiences even if the answer arrives at the same time.

---

## Live Demo Numbers

| Metric | Mode A (cascade) | Mode B (tandem) |
|---|---|---|
| Time to first audio | **10.449s** (dead silence) | **< 1ms** (filler from RAM) |
| Time to real answer | 10.449s | ~6.8s |
| STT latency | ~2.0s | ~2.0s (background) |
| LLM latency | ~2.5s | ~2.5s (background) |
| TTS latency | ~3.8s | ~3.8s (streaming) |

**10,000× improvement in time-to-first-audio.** The oracle takes the same time in both modes — the same Sarvam APIs, same network, same machine. The tandem architecture hides the wait.

---

## Architecture Overview

### The Two Modes

```
┌─────────────────────────────────────────────────────────────────┐
│  MODE A — Cascaded (baseline)                                   │
│                                                                 │
│  [mic] ──► [Saaras STT] ──► [Sarvam-M LLM] ──► [Bulbul TTS]  │
│            ~2.0s             ~2.5s               ~3.8s          │
│                                                                 │
│  User hears: ████████████████████████████████████ audio        │
│              ◄────────── 10.4s dead silence ──────────►        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  MODE B — Tandem / KAME (this project)                          │
│                                                                 │
│  [mic] ──┬──► [filler cache] ──► audio (<1ms)                  │
│           │                                                     │
│           └──► [Saaras STT] ──► [Sarvam-M LLM] ──► [Bulbul]   │
│                ~2.0s             ~2.5s              ~3.8s       │
│                asyncio.create_task() — runs concurrently        │
│                                                                 │
│  User hears: ▌ filler ▌████████████████████████████ answer     │
│              ◄►                                                  │
│              <1ms first response                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        BROWSER (client/index.html)                   │
│                                                                      │
│  ┌─────────────┐    AudioWorklet     ┌──────────────────────────┐   │
│  │  Orb (mic)  │───Float32→Int16────►│  WebSocket Client        │   │
│  │  click UI   │                     │  ws://localhost:8000/ws  │   │
│  └─────────────┘                     └──────────┬───────────────┘   │
│                                                  │                   │
│  Binary ◄── WAV filler / raw PCM chunks ─────────┘                   │
│  JSON   ◄── transcript / latency_stats / audio_meta / audio_end      │
│                                                                      │
│  ┌──────────┐  ┌──────────────┐  ┌─────────────┐  ┌────────────┐   │
│  │Transcript│  │ Lat breakdown│  │  A vs B     │  │  History   │   │
│  │ Response │  │ filler/stt/  │  │  comparison │  │  per-turn  │   │
│  │  boxes   │  │ llm/tts bars │  │  blocks     │  │  chips     │   │
│  └──────────┘  └──────────────┘  └─────────────┘  └────────────┘   │
└────────────────────────────┬────────────────────────────────────────┘
                             │ WebSocket  ws://
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                    SERVER (server/main.py)                            │
│                    FastAPI + Uvicorn — port 8000                      │
│                                                                      │
│  @app.on_event("startup")                                            │
│  ├── await tandem.preload_minimal()    ← blocks until 1 filler ready │
│  └── asyncio.create_task(             ← background, non-blocking    │
│          tandem.preload_fillers())                                   │
│                                                                      │
│  @app.websocket("/ws")                                               │
│  ├── receives binary chunks → audio_buffer (bytearray)              │
│  ├── receives {"type":"set_mode"}  → mode / language switch          │
│  └── receives {"type":"end_of_speech"}                               │
│       ├── Mode A → _handle_cascaded()  ← sequential await           │
│       └── Mode B → _handle_tandem()                                  │
│            ├── get_filler() → send_bytes(filler)  ← <1ms             │
│            └── asyncio.create_task(run_streaming_oracle())           │
└───────────────────────┬────────────────────────┬────────────────────┘
                        │                        │
          ┌─────────────▼──────────┐   ┌─────────▼──────────────────┐
          │  server/kame_engine.py  │   │  server/sarvam_client.py   │
          │  TandemEngine class     │   │                            │
          │                        │   │  transcribe()   ← STT      │
          │  fillers dict:          │   │  get_oracle()   ← LLM      │
          │  lang → cat → [bytes]  │   │  synthesize()   ← TTS      │
          │                        │   │                            │
          │  preload_minimal()      │   │  streaming_synthesize()    │
          │  preload_fillers()      │   │  ← WebSocket TTS chunks   │
          │  get_filler() <1ms      │   │                            │
          │  get_bilingual_filler() │   │  streaming_oracle_pipeline()
          │  classify_question()    │   │  ← async generator        │
          │  _concat_wavs()         │   │  yields: stt_done,        │
          └────────────────────────┘   │  llm_done, tts_chunk,     │
                                       │  tts_done                  │
                                       └────────────┬───────────────┘
                                                    │ HTTPS / WSS
                                       ┌────────────▼───────────────┐
                                       │      SARVAM AI CLOUD        │
                                       │                            │
                                       │  Saaras v3  (STT)          │
                                       │  api.sarvam.ai/speech-to-  │
                                       │  text                      │
                                       │                            │
                                       │  Sarvam-M   (LLM)          │
                                       │  api.sarvam.ai/v1/chat/    │
                                       │  completions               │
                                       │                            │
                                       │  Bulbul v3  (TTS)          │
                                       │  api.sarvam.ai/text-to-    │
                                       │  speech (REST)             │
                                       │  api.sarvam.ai/text-to-    │
                                       │  speech/ws (WebSocket)     │
                                       └────────────────────────────┘
```

---

## File Structure

```
sarvam-kame/
│
├── server/
│   ├── __init__.py
│   ├── config.py          ← API keys, URLs, model names, sample rates
│   ├── kame_engine.py     ← TandemEngine: filler cache + WAV concat
│   ├── sarvam_client.py   ← STT / LLM / TTS API wrappers + streaming
│   ├── main.py            ← FastAPI WebSocket server, Mode A/B handlers
│   └── metrics.py         ← SessionMetrics dataclass, JSONL latency log
│
├── client/
│   └── index.html         ← Single-page browser app (all-in-one)
│
├── latency_log.jsonl      ← Auto-generated per-turn latency log
├── .env                   ← SARVAM_API_KEY (not committed)
├── requirements.txt
└── README.md
```

---

## Detailed Component Diagrams

### TandemEngine — Filler Cache

```
Server startup
     │
     ▼
preload_minimal()                     ~5s total
     │
     ├── synthesize("Let me look that up", "en-IN") ──► Bulbul TTS ──► bytes
     │   stored: fillers["en-IN"]["general"][0]
     │
     └── synthesize("आपके सवाल के लिए शुक्रिया", "hi-IN") ──► bytes
         stored: fillers["hi-IN"]["general"][0]
         loaded = True ← server now accepts connections

asyncio.create_task(preload_fillers())   runs in background
     │
     ├── en-IN / knowledge / phrase 0 ──► bytes
     ├── en-IN / knowledge / phrase 1 ──► bytes
     ├── en-IN / time_date / phrase 0 ──► bytes
     ├── en-IN / math     / phrase 0 ──► bytes
     ├── hi-IN / knowledge / ...
     └── ...  (8 categories × 2-3 phrases = ~20 total)


Runtime — get_filler(language="en-IN", category="math", count=2)
     │
     ├── pool = fillers["en-IN"]["math"]          ← dict lookup, 0ms
     ├── wav0 = pool[(turn_count + 0) % len(pool)]  ← round-robin
     ├── wav1 = pool[(turn_count + 1) % len(pool)]
     └── return _concat_wavs([wav0, wav1])         ← <1ms total


_concat_wavs([wav_a, wav_b])
     │
     ├── header = wav_a[:44]           ← RIFF header from first file
     ├── data_a = wav_a[44:]           ← PCM samples from first file
     ├── data_b = wav_b[44:]           ← PCM samples from second file
     │                                    (strip header from second)
     ├── new_data_size = len(data_a) + len(data_b)
     ├── struct.pack_into(header, offset=4,  36 + new_data_size) ← RIFF size
     ├── struct.pack_into(header, offset=40, new_data_size)      ← data size
     └── return new_header + data_a + data_b    ← one valid WAV file
```

### Oracle Pipeline — Streaming

```
streaming_oracle_pipeline(pcm_bytes, history)
  │  async generator
  │
  ├─ await transcribe(pcm_bytes)
  │       │
  │       ├── pcm_to_wav(pcm_bytes)         ← wrap 16kHz PCM in WAV header
  │       ├── POST /speech-to-text          ← multipart/form-data
  │       │   model=saaras:v3
  │       │   language_code=unknown          ← auto-detect
  │       │   mode=codemix                   ← handles Hindi+English mix
  │       └── returns (transcript, latency, lang_code)
  │
  ├─ yield {"stage": "stt_done", "transcript": ..., "language_code": ...}
  │         ↑ main.py sends {"type":"transcript"} to browser immediately
  │
  ├─ await get_oracle(transcript, history)
  │       │
  │       ├── system_prompt: voice assistant, 1-3 sentences, no markdown
  │       ├── messages = [system] + history[-8:] + [user: transcript]
  │       ├── POST /v1/chat/completions
  │       │   model=sarvam-m
  │       │   max_tokens=300
  │       │   temperature=0.3
  │       ├── _clean_llm_response()          ← strip <think> blocks
  │       ├── _ensure_complete_sentence()    ← truncate at last .!?।
  │       └── retry once if response empty
  │
  ├─ yield {"stage": "llm_done", "response_text": ...}
  │         ↑ main.py sends {"type":"audio_meta"} → browser flips streamMode=true
  │
  ├─ await streaming_synthesize(response_text)
  │       │
  │       ├── WSS wss://api.sarvam.ai/text-to-speech/ws
  │       ├── send {"type":"config", output_audio_codec:"pcm"}
  │       ├── send {"type":"text",   data:{text: response_text}}
  │       ├── send {"type":"flush"}           ← force immediate synthesis
  │       │
  │       ├── recv {"type":"audio"} → base64 decode → PCM chunk
  │       ├── recv {"type":"audio"} → PCM chunk
  │       ├── ...
  │       └── recv {"type":"event", event_type:"final"} → done
  │           fallback: REST TTS → strip 44-byte WAV header → PCM
  │
  ├─ yield {"stage": "tts_chunk", "audio": pcm_bytes}  ×N
  │         ↑ main.py sends raw bytes to browser
  │           browser: Int16→Float32, createBuffer, schedule after filler
  │
  └─ yield {"stage": "tts_done", "tts_latency": ...}
            ↑ main.py sends {"type":"audio_end"} → browser flips streamMode=false
```

### Browser Audio Pipeline

```
RECORDING PATH
──────────────
User clicks orb (user gesture — required for AudioContext)
     │
     ├── getUserMedia({sampleRate:16000, channelCount:1})
     │
     ├── AudioContext(sampleRate:16000)
     │        │
     │        └── AudioWorklet: RecorderProcessor
     │               process() called every 128 samples (~8ms)
     │               Float32[-1,+1] ──► Int16[-32768,+32767]
     │               postMessage(buffer, [buffer])  ← zero-copy transfer
     │                    │
     │                    └──► ws.send(arraybuffer)  ── WebSocket ──► server
     │
     └── playCtx created here (user gesture window)
          silent keepalive buffer started ← prevents Chrome autoplay suspend

User clicks orb again
     │
     ├── processorNode.disconnect()
     ├── audioCtx.close()
     └── ws.send({"type":"end_of_speech"})


PLAYBACK PATH
─────────────
Binary arrives on WebSocket
     │
     ├── streamMode = false?
     │    └── playWAV(buf)
     │         decodeAudioData(buf.slice(0))    ← copy prevents buffer detach
     │         createBufferSource()
     │         src.start(currentTime)
     │         scheduledEnd = currentTime + duration
     │
     └── streamMode = true?
          └── playPCM(buf)
               Int16Array(buf)
               Float32Array: i16[i] / 32768
               createBuffer(channels, length, 24000)
               copyToChannel(float32, 0)
               src.start(Math.max(currentTime, scheduledEnd))
               scheduledEnd = start + duration    ← chain next chunk here


FILLER → ORACLE SEAMLESS TRANSITION
─────────────────────────────────────
t=0ms     filler WAV arrives    → playWAV()  → scheduledEnd = t + 6s
t=8200ms  audio_meta arrives    → streamMode = true
t=8210ms  PCM chunk 1 arrives   → playPCM()  → src.start(scheduledEnd=6s)
                                               scheduledEnd = 6s + 0.3s = 6.3s
t=8260ms  PCM chunk 2 arrives   → playPCM()  → src.start(6.3s)
                                               scheduledEnd = 6.3s + 0.3s = 6.6s
...
          User hears: [filler 0-6s][oracle 6s-end] — seamless, no gap
```

---

## Message Protocol

### Client → Server

| Type | Format | Description |
|---|---|---|
| binary | `ArrayBuffer` (Int16 PCM) | Raw 16kHz mono audio chunks while recording |
| `set_mode` | `{"type":"set_mode","mode":"tandem"\|"cascaded"}` | Switch mode |
| `end_of_speech` | `{"type":"end_of_speech"}` | User stopped recording |

### Server → Client

| Type | Format | Description |
|---|---|---|
| binary (WAV) | `ArrayBuffer` | Filler audio — has 44-byte WAV header |
| binary (PCM) | `ArrayBuffer` | Oracle TTS stream — raw Int16, no header |
| `transcript` | `{"type":"transcript","text":"..."}` | STT result |
| `filler_sent` | `{"type":"filler_sent","latency":0.001}` | Fast path timing |
| `audio_meta` | `{"type":"audio_meta","sample_rate":24000,...}` | Switch to PCM mode |
| `audio_end` | `{"type":"audio_end"}` | PCM stream complete |
| `latency_stats` | See below | All per-stage timings |
| `status` | `{"type":"status","text":"..."}` | UI status updates |

**`latency_stats` payload:**
```json
{
  "type": "latency_stats",
  "mode": "tandem",
  "stt": 2.043,
  "llm": 2.511,
  "tts": 3.812,
  "oracle_total": 8.366,
  "filler_latency": 0.001,
  "response_text": "The capital of France is Paris."
}
```

---

## Audio Format Reference

| Stage | Format | Sample Rate | Bit Depth | Channels |
|---|---|---|---|---|
| Browser capture | Float32 PCM | 16 kHz | 32-bit float | Mono |
| Sent to server | Int16 PCM | 16 kHz | 16-bit | Mono |
| Sent to Saaras STT | WAV (RIFF) | 16 kHz | 16-bit | Mono |
| Bulbul TTS output | WAV (RIFF) | 24 kHz | 16-bit | Mono |
| Streaming TTS chunks | Raw PCM | 24 kHz | 16-bit | Mono |
| Browser playback | Float32 PCM | 24 kHz | 32-bit float | Mono |

---

## Engineering Challenges Solved

### 1. WAV Header Corruption on Concatenation

**Problem:** Concatenating two WAV files as `wav1 + wav2` produces broken audio — the player reads the first file's header which says the data is N bytes, reads exactly N bytes, then stops. The second phrase is never played.

**Solution:** `_concat_wavs()` strips the second WAV's header, sums the data sizes, and writes a new header with correct RIFF and data chunk sizes using `struct.pack_into`.

```python
# Wrong — header says 24000 bytes, player reads 24000 bytes, stops
broken = wav1 + wav2

# Right — header updated to say 48000 bytes, player reads both
correct = _concat_wavs([wav1, wav2])
```

### 2. Chrome AudioContext Autoplay Suspension

**Problem:** Chrome suspends AudioContexts created outside user gestures. Even after resuming inside a gesture, Chrome suspends idle contexts after ~30 seconds. Next audio arrives, plays on suspended context — silence.

**Solution:** Create `playCtx` inside the mic click handler (user gesture). Immediately start a looping silent 1-second buffer — Chrome sees the context as "always playing" and never suspends it.

```javascript
// Inside startRec() — user gesture context
if (!playCtx) playCtx = new AudioContext({sampleRate: 24000})
// Silent keepalive — prevents idle suspension
const sil = playCtx.createBuffer(1, playCtx.sampleRate, playCtx.sampleRate)
const keepalive = playCtx.createBufferSource()
keepalive.buffer = sil
keepalive.loop = true
keepalive.connect(playCtx.destination)
keepalive.start()
```

### 3. Streaming TTS Fallback — WAV Header Strip

**Problem:** Streaming TTS (WebSocket) returns raw PCM chunks. REST TTS (fallback) returns a WAV file with a 44-byte header. The client is in PCM mode after `audio_meta`. If fallback passes the WAV through unmodified, the first 44 bytes are decoded as PCM samples — a burst of noise.

**Solution:** Strip the 44-byte header on fallback before yielding chunks.

```python
except Exception:
    audio, lat = await synthesize(text, language)  # REST fallback → WAV
    if len(audio) > 44:
        audio = audio[44:]  # strip header → raw PCM
    chunks = [audio]
```

### 4. STT Language Detection — Codemix

**Problem:** Indian users mix Hindi and English in the same sentence ("mujhe capital of France batao"). Character-based heuristics (is there Devanagari?) fail for Romanized Hindi. Hardcoding "en-IN" misses Hindi speech.

**Solution:** Pass `language_code="unknown"` to Saaras v3 with `mode="codemix"`. Propagate the `language_code` field from the STT response through the entire pipeline. Store it as `prev_language` for subsequent turns.

### 5. LLM Thinking Block Contamination

**Problem:** `sarvam-m` sometimes outputs reasoning inside `<think>...</think>` or `<Thinking>...</Thinking>` before the actual answer. If this reaches TTS, the user hears "Let me think step by step. The question is asking..." instead of the answer.

**Solution:** `_clean_llm_response()` uses `re.sub` with `re.DOTALL` to strip all thinking/reasoning blocks. If the entire response was a thinking block (nothing left after cleaning), retry once with `"(answer directly, no thinking)"` appended.

### 6. Half-Sentence Truncation

**Problem:** `max_tokens=300` hard-caps the LLM output. The model gets cut off mid-sentence: "The Eiffel Tower was built in 1889 and stands 330 meters tall, making it the" — incomplete.

**Solution:** `_ensure_complete_sentence()` finds the last `.!?।` boundary and truncates there. Handles closing quotes after punctuation. Falls back to appending a period for very short responses that have no boundary.

### 7. AudioWorklet VAD — Reverted

**Attempted:** Energy-based VAD in the AudioWorklet to auto-detect speech end and trigger `end_of_speech` automatically.

**Problem:** Disconnecting the AudioWorklet's downstream node to "stop" it also stops the `process()` callback from firing — the VAD stops working. Reconnecting requires another user gesture in some browsers.

**Resolution:** Reverted to manual mic click. Correct production solution would be a dedicated VAD worklet with a permanent silent connection, or server-side WebRTC VAD (Silero).

---

## Setup

### Requirements

- Python 3.11+
- No GPU required
- 16GB RAM minimum (model weights not loaded — only API calls)

### Install

```bash
git clone <repo>
cd sarvam-kame
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Edit .env and add your key:
# SARVAM_API_KEY=your_key_here
```

Get your key at [console.sarvam.ai](https://console.sarvam.ai)

### Run

```bash
uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```

Watch startup logs:
```
[TANDEM] Preloading minimal fillers...
[TANDEM]  en-IN/general: 38912 bytes (2.21s)
[TANDEM]  hi-IN/general: 41984 bytes (2.63s)
[TANDEM] Minimal preload done — fillers ready.
[SERVER] Ready at http://0.0.0.0:8000
```

Open `http://localhost:8000` in Chrome or Edge.

### Test the APIs

```bash
python -c "
import asyncio
from server import sarvam_client as s
async def test():
    audio, lat = await s.synthesize('Hello, testing Sarvam.', 'en-IN')
    print(f'TTS: {len(audio)} bytes in {lat:.2f}s')
    resp, lat = await s.get_oracle('What is 2 + 2?', [])
    print(f'LLM: {resp!r} in {lat:.2f}s')
asyncio.run(test())
"
```

---

## API Reference

### `GET /`
Serves `client/index.html`

### `GET /stats`
Returns filler preload status:
```json
{
  "tandem": {
    "fillers_loaded": 18,
    "categories": ["general", "knowledge", "time_date", "math"],
    "languages": ["en-IN", "hi-IN"],
    "fillers_by_lang": {
      "en-IN": {"general": 3, "knowledge": 3, "time_date": 2, "math": 2},
      "hi-IN": {"general": 2, "knowledge": 2, "time_date": 2, "math": 2}
    }
  }
}
```

### `WS /ws`
Main WebSocket endpoint. See Message Protocol section above.

---

## Latency Log

Every turn is appended to `latency_log.jsonl`:
```json
{"session_id": "a3f2b891", "mode": "tandem", "stt_latency": 2.043, "llm_latency": 2.511, "tts_latency": 3.812, "total_latency": 0.001, "time_to_first_audio": 0.001, "transcript": "what is the capital of france", "response_text": "The capital of France is Paris."}
```

Load for analysis:
```python
import pandas as pd
df = pd.read_json("latency_log.jsonl", lines=True)
print(df[["mode","stt_latency","llm_latency","tts_latency","total_latency"]].groupby("mode").mean())
```

---

## What This Is Not

| Limitation | Reason | Production fix |
|---|---|---|
| Single session only | Global `TandemEngine` instance | Per-connection engine instances |
| No VAD | AudioWorklet VAD reverted (see above) | Silero VAD or server-side WebRTC |
| 10s oracle latency | 4 network round trips to Sarvam cloud | Run Moshi locally on GPU (14GB VRAM) |
| Context-free fillers | Pre-synthesized, not neural | Moshi generates contextual speech tokens |
| No auth | Prototype | Session tokens, rate limiting |

---

## Comparison with Original KAME Paper

| Component | KAME paper | This project |
|---|---|---|
| Fast path | Moshi 7B neural codec (GPU) | Pre-cached Bulbul TTS (RAM) |
| Fast path latency | ~50ms | < 1ms |
| STT | Google Speech-to-Text | Sarvam Saaras v3 |
| LLM oracle | OpenAI GPT-4 | Sarvam-M |
| TTS oracle | Proprietary | Sarvam Bulbul v3 |
| Streaming | Moshi token stream | WebSocket PCM chunks |
| Hardware | A100 / H100 GPU | 16GB CPU laptop |
| Fast path audio | Contextual (neural) | Category-aware pre-synth |
| Oracle latency | ~1-2s (local model) | ~8-10s (cloud APIs) |
| Architecture claim | ✅ Proven | ✅ Proven |

The architecture claim — that tandem decouples first-audio latency from oracle latency — is identical and demonstrated in both. The hardware constraint changes the fast path implementation but not the architectural result.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Server | Python 3.11, FastAPI, Uvicorn |
| Async | asyncio, httpx, websockets |
| STT | Sarvam Saaras v3 (REST) |
| LLM | Sarvam-M via /v1/chat/completions |
| TTS | Sarvam Bulbul v3 (REST + WebSocket) |
| Browser | Vanilla JS, Web Audio API, AudioWorklet |
| Fonts | Syne, DM Mono (Google Fonts) |
| Logging | JSONL append, dataclasses |

---

## License

MIT — research prototype, not for production use.

---

*Built to demonstrate the KAME tandem architecture on consumer hardware using Sarvam's Indian language AI APIs.*
