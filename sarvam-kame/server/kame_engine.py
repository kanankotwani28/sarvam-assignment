"""
Wraps the KAME/Moshi model for CPU inference.
On CPU: ~300-600ms per 80ms chunk (slower than real-time).
On GPU: ~80ms per chunk (real-time).

We load the model once at startup and reuse it per session.
The engine is NOT thread-safe — one session at a time.
"""

import time
import numpy as np
import torch
from typing import Optional


class KameEngine:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.mimi = None
        self.lm_gen = None
        self.text_tokenizer = None
        self.loaded = False
        self._session_active = False
        self._input_buf = np.array([], dtype=np.float32)

    def load(self, hf_repo: str = "SakanaAI/kame"):
        """
        Load KAME checkpoint from local HF cache.
        Uses local cached files to avoid HF network calls.
        """
        print(f"[KAME] Loading from {hf_repo} on {self.device}...")

        try:
            from moshi.models import loaders
            from moshi.models.lm import LMGen
            from pathlib import Path

            # Locate cached files in HF cache
            cache_root = Path.home() / ".cache" / "huggingface" / "hub"
            kame_dir = cache_root / f"models--SakanaAI--kame" / "snapshots"
            kyutai_dir = cache_root / f"models--kyutai--moshiko-pytorch-bf16" / "snapshots"

            kame_snapshot = next(kame_dir.iterdir())
            kyutai_snapshot = next(kyutai_dir.iterdir())

            config_path = kame_snapshot / "config.json"
            moshi_weights = kame_snapshot / "model.safetensors"
            mimi_weights = kyutai_snapshot / "tokenizer-e351c8d8-checkpoint125.safetensors"
            tokenizer_path = kyutai_snapshot / "tokenizer_spm_32k_3.model"

            checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
                hf_repo,
                moshi_weights=moshi_weights,
                mimi_weights=mimi_weights,
                tokenizer=tokenizer_path,
                config_path=config_path,
            )

            # Load Mimi codec
            self.mimi = checkpoint_info.get_mimi(device=self.device)
            # Load text tokenizer (for decoding generated text tokens)
            self.text_tokenizer = checkpoint_info.get_text_tokenizer()
            # Load the LMModel (weights ~31GB)
            lm = checkpoint_info.get_moshi(device=self.device)

            # KAME has no conditioners, so condition_tensors is empty
            condition_tensors = {}
            self.lm_gen = LMGen(
                lm,
                cfg_coef=1.0,
                condition_tensors=condition_tensors,
                use_sampling=False,
                temp=0.8,
                temp_text=0.7,
                top_k=250,
                **checkpoint_info.lm_gen_config,
            )

            # Start streaming — keeps state alive across sessions
            self.mimi.streaming_forever(1)
            self.lm_gen.streaming_forever(1)

            # 1920 samples at 24kHz (80ms per Mimi frame)
            self._frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
            # 1280 samples at 16kHz (80ms, matching client AudioWorklet)
            self._frame_16k = 1280

            self.loaded = True
            print(f"[KAME] Model loaded! frame_size={self._frame_size}")

        except Exception as e:
            print(f"[KAME] Load failed: {e}")
            import traceback
            traceback.print_exc()
            print("[KAME] Running in Sarvam-only fallback mode.")
            self.loaded = False

    def start_session(self):
        """Reset model state for a new conversation session."""
        if not self.loaded:
            return
        self._session_active = True
        self._input_buf = np.array([], dtype=np.float32)
        self.lm_gen.reset_streaming()
        self.mimi.reset_streaming()

    def end_session(self):
        self._session_active = False
        self._input_buf = np.array([], dtype=np.float32)

    def step(self, pcm_chunk: bytes, oracle_tokens=None) -> bytes:
        """
        Process incoming 16kHz PCM through KAME streaming inference.

        Accumulates 16kHz audio frames, resamples to 24kHz on the fly,
        feeds one 80ms frame at a time through Mimi encode → LMGen step → Mimi decode.

        pcm_chunk:    raw 16-bit PCM at 16kHz (from browser AudioWorklet)
        oracle_tokens: ignored — KAME does not support external text conditioning
                       without model-level changes. The oracle pipeline in main.py
                       still sends TTS audio directly to the client.
        Returns: raw 16-bit PCM audio bytes at 24kHz (browser plays at 24kHz)
        """
        if not self.loaded or not self._session_active:
            return b""

        try:
            # 1. Accumulate incoming 16kHz float32 audio
            chunk = np.frombuffer(pcm_chunk, dtype=np.int16).astype(np.float32) / 32768.0
            self._input_buf = np.concatenate([self._input_buf, chunk])

            output_chunks = []

            # 2. Consume one 80ms frame at a time
            while len(self._input_buf) >= self._frame_16k:
                frame_16k = self._input_buf[:self._frame_16k]
                self._input_buf = self._input_buf[self._frame_16k:]

                t = time.perf_counter()

                # 3. Resample 16kHz → 24kHz (1280 → 1920 samples)
                frame_24k = self._resample_16k_to_24k(frame_16k)

                # 4. Mimi encode → LMGen step → Mimi decode
                audio = torch.from_numpy(frame_24k).unsqueeze(0).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    codes = self.mimi.encode(audio)           # [1, 8, 1]
                    tokens = self.lm_gen.step(codes)          # [1, 9, 1] or None

                    if tokens is None:
                        # First frame returns None while delays fill up
                        continue

                    # tokens[:, 0] = text token, tokens[:, 1:] = 8 depformer audio codes
                    out_audio = self.mimi.decode(tokens[:, 1:])  # [1, 1, 1920]

                elapsed = time.perf_counter() - t
                print(f"[KAME] step: {elapsed*1000:.0f}ms", end="\r")
                output_chunks.append(out_audio.cpu())

            if not output_chunks:
                return b""

            # 5. Concatenate output frames and convert to bytes
            out = torch.cat(output_chunks, dim=2)
            out_np = out.squeeze().numpy()
            out_int16 = (out_np * 32768).clip(-32768, 32767).astype(np.int16)
            return out_int16.tobytes()

        except Exception as e:
            print(f"[KAME] step error: {e}")
            import traceback
            traceback.print_exc()
            return b""

    @staticmethod
    def _resample_16k_to_24k(audio_16k: np.ndarray) -> np.ndarray:
        """Linear interpolation resample from 16kHz to 24kHz."""
        n_in = len(audio_16k)
        n_out = int(n_in * 24000 / 16000)
        x_in = np.arange(n_in)
        x_out = np.linspace(0, n_in - 1, n_out)
        return np.interp(x_out, x_in, audio_16k)


# ── Global singleton, loaded once at server startup ──
kame = KameEngine(device="cpu")
