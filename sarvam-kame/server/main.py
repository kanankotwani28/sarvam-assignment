import asyncio
import json
import time
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from .config import HOST, PORT
from .sarvam_client import oracle_pipeline, streaming_oracle_pipeline
from .kame_engine import tandem, classify_question
from .metrics import SessionMetrics

app = FastAPI(title="Sarvam-KAME Voice Agent")


@app.get("/")
async def index():
    return HTMLResponse(Path("client/index.html").read_text(encoding="utf-8"))


@app.get("/stats")
async def stats():
    return {
        "tandem": tandem.get_filler_stats(),
    }


@app.on_event("startup")
async def startup():
    # Block until minimal fillers are ready — ensures no silent filler
    await tandem.preload_minimal()
    # Full preload in background (categories + extra languages)
    asyncio.create_task(tandem.preload_fillers())
    print(f"[SERVER] Ready at http://{HOST}:{PORT}")


# ════════════════════════════════════════════
#  SINGLE WEBSOCKET ENDPOINT
#  Client sends mode in text messages:
#    {"type": "set_mode", "mode": "cascaded"|"tandem"}
#    {"type": "end_of_speech"}
#  Binary messages = raw 16kHz 16-bit PCM audio chunks
# ════════════════════════════════════════════
@app.websocket("/ws")
async def ws_handler(websocket: WebSocket):
    await websocket.accept()
    mode = "tandem"  # default
    history: list[dict] = []
    audio_buffer = bytearray()
    m = SessionMetrics(mode=mode)
    prev_category = None
    prev_language = "en-IN"

    print(f"[WS] Client connected (default mode: {mode})")

    async def handle_end_of_speech():
        nonlocal m
        m = SessionMetrics(mode=mode)
        m.t_speech_end = time.perf_counter()

        pcm_data = bytes(audio_buffer)
        audio_buffer.clear()

        if len(pcm_data) < 3200:
            await websocket.send_text(json.dumps(
                {"type": "status", "text": "Audio too short. Try speaking longer."}
            ))
            return

        if mode == "cascaded":
            await _handle_cascaded(pcm_data, history, websocket, m)
        else:
            await _handle_tandem(pcm_data, history, websocket, m)

    async def _handle_cascaded(pcm_data, history, ws, m):
        nonlocal prev_category
        """Mode A: sequential STT → LLM → TTS, audio at the end."""
        await ws.send_text(json.dumps({"type": "status", "text": "Transcribing..."}))
        result = await oracle_pipeline(pcm_data, history)
        m.t_stt_done = m.t_speech_end + result["stt_latency"]
        m.t_llm_done = m.t_stt_done + result["llm_latency"]
        m.t_tts_done = m.t_llm_done + result["tts_latency"]
        m.transcript = result["transcript"]
        m.response_text = result["response_text"]

        if result["transcript"]:
            await ws.send_text(json.dumps({
                "type": "transcript", "text": result["transcript"]
            }))
            prev_category = classify_question(result["transcript"])

        if not result["response_text"]:
            await ws.send_text(json.dumps({"type": "status", "text": "Ready"}))
            return

        if result["audio_bytes"]:
            m.t_first_audio_sent = time.perf_counter()
            await ws.send_bytes(result["audio_bytes"])

        await ws.send_text(json.dumps({
            "type": "latency_stats",
            "mode": "cascaded",
            "stt":   round(result["stt_latency"], 3),
            "llm":   round(result["llm_latency"], 3),
            "tts":   round(result["tts_latency"], 3),
            "total": round(m.total_latency or result["total_latency"], 3),
            "response_text": result["response_text"],
        }))

        await ws.send_text(json.dumps({"type": "status", "text": "Ready"}))

        m.log()
        history.extend([
            {"role": "user",      "content": result["transcript"]},
            {"role": "assistant", "content": result["response_text"]},
        ])
        if len(history) > 20:
            history[:] = history[-20:]

    async def _handle_tandem(pcm_data, history, ws, m):
        nonlocal prev_category, prev_language
        """
        Mode B: send filler immediately, then streaming background oracle.
        Filler plays locally; TTS PCM chunks stream in as Sarvam generates them.
        """
        # ── Fast path: send filler immediately ──
        filler = tandem.get_filler(language=prev_language, category=prev_category)
        filler_time = time.perf_counter()
        m.t_first_audio_sent = filler_time
        m.t_speech_start = m.t_speech_end

        if filler:
            ttfa = m.time_to_first_audio or 0.0
            print(f"[TANDEM] Filler sent: {len(filler)} bytes, TTFA={ttfa:.3f}s")
            await ws.send_text(json.dumps({
                "type": "filler_sent",
                "latency": round(ttfa, 3),
            }))
            await ws.send_bytes(filler)

        # ── Streaming oracle in background ──
        async def run_streaming_oracle():
            nonlocal prev_category, prev_language
            stt_lat = llm_lat = tts_lat = 0.0
            transcript = response_text = ""

            async for event in streaming_oracle_pipeline(pcm_data, history):
                stage = event["stage"]

                if stage == "stt_done":
                    stt_lat = event["stt_latency"]
                    m.t_stt_done = m.t_speech_end + stt_lat
                    transcript = event["transcript"]
                    m.transcript = transcript

                    if transcript:
                        await ws.send_text(json.dumps({
                            "type": "transcript",
                            "text": transcript,
                        }))
                        prev_category = classify_question(transcript)
                        prev_language = "hi-IN" if any(
                            '\u0900' <= c <= '\u097F' for c in transcript
                        ) else "en-IN"

                elif stage == "llm_done":
                    llm_lat = event["llm_latency"]
                    m.t_llm_done = m.t_stt_done + llm_lat
                    response_text = event["response_text"]
                    m.response_text = response_text

                    # Tell client to expect raw PCM stream at 24 kHz
                    await ws.send_text(json.dumps({
                        "type": "audio_meta",
                        "sample_rate": 24000,
                        "channels": 1,
                        "bits": 16,
                        "format": "pcm_s16le",
                    }))

                elif stage == "tts_chunk":
                    m.t_first_audio_sent = m.t_first_audio_sent or time.perf_counter()
                    await ws.send_bytes(event["audio"])

                elif stage == "tts_done":
                    tts_lat = event["tts_latency"]
                    m.t_tts_done = m.t_llm_done + tts_lat
                    await ws.send_text(json.dumps({"type": "audio_end"}))

            await ws.send_text(json.dumps({
                "type": "latency_stats",
                "mode": "tandem",
                "stt":            round(stt_lat, 3),
                "llm":            round(llm_lat, 3),
                "tts":            round(tts_lat, 3),
                "oracle_total":   round(stt_lat + llm_lat + tts_lat, 3),
                "filler_latency": round(ttfa, 3),
                "response_text":  response_text,
            }))

            await ws.send_text(json.dumps({"type": "status", "text": "Ready"}))

            m.log()
            if transcript and response_text:
                history.extend([
                    {"role": "user",      "content": transcript},
                    {"role": "assistant", "content": response_text},
                ])
                if len(history) > 20:
                    history[:] = history[-20:]

        asyncio.create_task(run_streaming_oracle())

    try:
        while True:
            data = await websocket.receive()

            if "bytes" in data:
                audio_buffer.extend(data["bytes"])

            elif "text" in data:
                msg = json.loads(data["text"])

                if msg.get("type") == "set_mode":
                    mode = msg["mode"]
                    print(f"[WS] Mode → {mode}")
                    await websocket.send_text(json.dumps({
                        "type": "mode_set", "mode": mode
                    }))

                elif msg.get("type") == "end_of_speech":
                    await handle_end_of_speech()

    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        print(f"[WS] Error: {e}")
        import traceback; traceback.print_exc()
