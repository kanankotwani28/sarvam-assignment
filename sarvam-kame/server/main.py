import asyncio
import json
import time
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from .config import HOST, PORT
from .sarvam_client import transcribe, get_oracle, synthesize
from .kame_engine import kame
from .metrics import SessionMetrics

app = FastAPI(title="Sarvam-KAME Voice Agent")

# ── Serve browser UI ─────────────────────────
@app.get("/")
async def index():
    return HTMLResponse(Path("client/index.html").read_text(encoding="utf-8"))

# ── Load KAME model on startup (non-blocking) ────────────────
@app.on_event("startup")
async def startup():
    print(f"[SERVER] Ready at http://{HOST}:{PORT}")
    # Load KAME in background thread (will download ~4GB on first run)
    def load_kame():
        import time
        print("[SERVER] Loading KAME in background (first run downloads ~4GB)...")
        kame.load()
        print(f"[SERVER] KAME loaded: {kame.loaded}")
    import threading
    thread = threading.Thread(target=load_kame, daemon=True)
    thread.start()


# ════════════════════════════════════════════
#  MODE A — CASCADED BASELINE
#  user finishes → STT → LLM → TTS → audio
# ════════════════════════════════════════════
@app.websocket("/ws/cascaded")
async def ws_cascaded(websocket: WebSocket):
    await websocket.accept()
    history: list[dict] = []
    audio_buffer = bytearray()
    print("[CASCADED] Client connected")

    try:
        while True:
            data = await websocket.receive()

            if "bytes" in data:
                audio_buffer.extend(data["bytes"])

            elif "text" in data:
                msg = json.loads(data["text"])

                if msg.get("type") == "end_of_speech":
                    m = SessionMetrics(mode="cascaded")
                    m.t_speech_end = time.perf_counter()

                    if len(audio_buffer) < 3200:
                        audio_buffer.clear()
                        continue

                    # ── Step 1: STT ──
                    await websocket.send_text(json.dumps(
                        {"type": "status", "text": "Transcribing..."}
                    ))
                    transcript, stt_lat = await transcribe(bytes(audio_buffer))
                    m.t_stt_done = time.perf_counter()
                    m.transcript = transcript
                    audio_buffer.clear()

                    if not transcript:
                        await websocket.send_text(json.dumps(
                            {"type": "status", "text": "Could not transcribe. Try again."}
                        ))
                        continue

                    await websocket.send_text(json.dumps({
                        "type": "transcript", "text": transcript
                    }))

                    # ── Step 2: LLM ──
                    await websocket.send_text(json.dumps(
                        {"type": "status", "text": "Thinking..."}
                    ))
                    response, llm_lat = await get_oracle(transcript, history)
                    m.t_llm_done = time.perf_counter()
                    m.response_text = response

                    if not response:
                        continue

                    # ── Step 3: TTS ──
                    await websocket.send_text(json.dumps(
                        {"type": "status", "text": "Generating speech..."}
                    ))
                    audio_bytes, tts_lat = await synthesize(response)
                    m.t_tts_done = time.perf_counter()

                    # ── Send audio ──
                    if audio_bytes:
                        m.t_first_audio_sent = time.perf_counter()
                        await websocket.send_bytes(audio_bytes)

                    # ── Send stats to UI ──
                    await websocket.send_text(json.dumps({
                        "type": "latency_stats",
                        "mode": "cascaded",
                        "stt":   round(stt_lat,  3),
                        "llm":   round(llm_lat,  3),
                        "tts":   round(tts_lat,  3),
                        "total": round(m.total_latency or 0, 3),
                        "response_text": response,
                    }))

                    await websocket.send_text(json.dumps(
                        {"type": "status", "text": "Ready"}
                    ))

                    # Log and update history
                    m.log()
                    history.extend([
                        {"role": "user",      "content": transcript},
                        {"role": "assistant", "content": response},
                    ])
                    if len(history) > 20:
                        history = history[-20:]

    except WebSocketDisconnect:
        print("[CASCADED] Client disconnected")
    except Exception as e:
        print(f"[CASCADED] Error: {e}")
        import traceback; traceback.print_exc()


# ════════════════════════════════════════════
#  MODE B — KAME TANDEM
#  Moshi responds immediately while
#  Sarvam STT+LLM runs in parallel as oracle
# ════════════════════════════════════════════
@app.websocket("/ws/kame")
async def ws_kame(websocket: WebSocket):
    await websocket.accept()
    history: list[dict]     = []
    audio_buffer            = bytearray()
    oracle_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    m = SessionMetrics(mode="kame")

    print("[KAME] Client connected")
    kame.start_session()

    async def oracle_pipeline(pcm_data: bytes, turn_start: float):
        """
        Runs in background: STT → LLM → TTS → send oracle audio.
        Does NOT block Moshi's inference loop.
        """
        nonlocal m
        m = SessionMetrics(mode="kame")
        m.t_speech_end = turn_start

        # STT
        await websocket.send_text(json.dumps(
            {"type": "status", "text": "Oracle: transcribing..."}
        ))
        transcript, stt_lat = await transcribe(pcm_data)
        m.t_stt_done = time.perf_counter()
        m.transcript = transcript

        if not transcript:
            return

        await websocket.send_text(json.dumps({
            "type": "transcript", "text": transcript
        }))

        # LLM oracle
        await websocket.send_text(json.dumps(
            {"type": "status", "text": "Oracle: generating response..."}
        ))
        response, llm_lat = await get_oracle(transcript, history)
        m.t_llm_done = time.perf_counter()
        m.response_text = response

        if not response:
            return

        # Drop oracle into queue — Moshi picks it up (last-write-wins)
        if not oracle_queue.empty():
            try:
                oracle_queue.get_nowait()   # discard stale oracle
            except asyncio.QueueEmpty:
                pass
        await oracle_queue.put(response)

        # TTS — high-quality oracle audio sent as "correction"
        oracle_audio, tts_lat = await synthesize(response)
        m.t_tts_done = time.perf_counter()

        if oracle_audio:
            await websocket.send_bytes(oracle_audio)

        # Stats
        await websocket.send_text(json.dumps({
            "type": "latency_stats",
            "mode": "kame",
            "stt":            round(stt_lat,  3),
            "llm":            round(llm_lat,  3),
            "tts":            round(tts_lat,  3),
            "oracle_total":   round((m.t_tts_done - m.t_speech_end), 3),
            "time_to_first":  round(m.time_to_first_audio or 0, 3),
            "response_text":  response,
        }))

        m.log()
        history.extend([
            {"role": "user",      "content": transcript},
            {"role": "assistant", "content": response},
        ])
        if len(history) > 20:
            history = history[-20:]

    try:
        while True:
            data = await websocket.receive()

            if "bytes" in data:
                chunk = data["bytes"]
                audio_buffer.extend(chunk)

                # ── Moshi immediate response (fast path) ──
                if kame.loaded:
                    if m.t_speech_start is None:
                        m.t_speech_start = time.perf_counter()

                    # Get latest oracle if available (non-blocking)
                    oracle_text = None
                    try:
                        oracle_text = oracle_queue.get_nowait()
                        print(f"[KAME] Oracle injected: '{oracle_text[:40]}...'")
                    except asyncio.QueueEmpty:
                        pass

                    # Run one Moshi step
                    immediate_audio = await asyncio.get_event_loop().run_in_executor(
                        None, kame.step, chunk, oracle_text
                    )

                    if immediate_audio:
                        if m.t_first_audio_sent is None:
                            m.t_first_audio_sent = time.perf_counter()
                            ttfa = m.time_to_first_audio
                            print(f"[KAME] First audio: {ttfa:.3f}s after speech start")
                            await websocket.send_text(json.dumps({
                                "type": "first_audio",
                                "latency": round(ttfa, 3)
                            }))
                        await websocket.send_bytes(immediate_audio)

            elif "text" in data:
                msg = json.loads(data["text"])

                if msg.get("type") == "end_of_speech":
                    turn_start = time.perf_counter()
                    m.t_speech_start = None  # reset for next turn
                    m.t_first_audio_sent = None

                    pcm_snapshot = bytes(audio_buffer)
                    audio_buffer.clear()

                    if len(pcm_snapshot) > 3200:
                        # Launch oracle in background — non-blocking
                        asyncio.create_task(
                            oracle_pipeline(pcm_snapshot, turn_start)
                        )

    except WebSocketDisconnect:
        print("[KAME] Client disconnected")
        kame.end_session()
    except Exception as e:
        print(f"[KAME] Error: {e}")
        import traceback; traceback.print_exc()
        kame.end_session()