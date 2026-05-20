"""
Wraps the KAME/Moshi model for CPU inference.
On CPU: ~300-600ms per 80ms chunk (slower than real-time).
On GPU: ~80ms per chunk (real-time).

We load the model once at startup and reuse it per session.
The engine is NOT thread-safe — one session at a time.
"""

import torch
import time
import io
import struct
from typing import Optional

class KameEngine:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None
        self.mimi  = None
        self.loaded = False
        self._session_active = False

    def load(self, hf_repo: str = "SakanaAI/kame"):
        """
        Download and load KAME checkpoint.
        First run: downloads ~3-4GB to ~/.cache/huggingface/
        Subsequent runs: loads from cache, ~30 seconds on CPU.
        """
        print(f"[KAME] Loading from {hf_repo} on {self.device}...")
        print("[KAME] First run downloads ~4GB — please wait...")

        try:
            # Step 1: get the moshi package's loader
            from moshi.models import loaders

            # Step 2: download config via huggingface_hub
            import huggingface_hub as hf
            
            # Download all required files
            local_dir = hf.snapshot_download(
                repo_id=hf_repo,
                ignore_patterns=["*.md", "*.txt"]
            )
            print(f"[KAME] Downloaded to: {local_dir}")

            # Step 3: load model using moshi's API
            # Check what loaders are available in your installed version:
            # python -c "from moshi.models import loaders; print(dir(loaders))"
            import os
            config_path = os.path.join(local_dir, "config.json")
            
            if hasattr(loaders, 'get_moshi_lm'):
                self.model, self.mimi = loaders.get_moshi_lm(
                    config_path, device=self.device
                )
            elif hasattr(loaders, 'get_lm_model'):
                self.model, self.mimi = loaders.get_lm_model(
                    config_path, device=self.device
                )
            else:
                # Fallback: list available functions
                available = [x for x in dir(loaders) if not x.startswith('_')]
                print(f"[KAME] Available loaders: {available}")
                raise RuntimeError(
                    f"Cannot find loader. Available: {available}\n"
                    "Check KAME/moshi GitHub for the correct loader function name."
                )

            self.model.eval()
            self.loaded = True
            print("[KAME] Model loaded successfully!")

        except Exception as e:
            print(f"[KAME] Load failed: {e}")
            print("[KAME] Running in Sarvam-only fallback mode.")
            self.loaded = False

    def start_session(self):
        """Reset model state for a new conversation session."""
        if not self.loaded:
            return
        self._session_active = True
        # Reset any cached state in the model
        if hasattr(self.model, 'reset_streaming'):
            self.model.reset_streaming()

    def end_session(self):
        self._session_active = False

    def step(self, pcm_chunk: bytes, oracle_tokens=None) -> bytes:
        """
        Process one ~80ms chunk of PCM audio through Moshi.
        Returns PCM audio bytes to play immediately.

        pcm_chunk: raw 16-bit PCM at 24kHz (Moshi's internal rate)
        oracle_tokens: optional tensor of oracle token ids from LLM
        Returns: raw 16-bit PCM audio bytes
        """
        if not self.loaded or not self._session_active:
            return b""

        try:
            import numpy as np
            t = time.perf_counter()

            # Convert bytes → float32 tensor
            audio_np = np.frombuffer(pcm_chunk, dtype=np.int16).astype(np.float32)
            audio_np /= 32768.0
            audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0)
            audio_tensor = audio_tensor.to(self.device)

            with torch.no_grad():
                # Encode input audio to tokens using Mimi codec
                input_tokens = self.mimi.encode(audio_tensor)

                # Run one step of the S2S transformer
                # oracle_tokens injected here if available
                if oracle_tokens is not None and hasattr(self.model, 'step_with_oracle'):
                    output_tokens = self.model.step_with_oracle(
                        input_tokens, oracle_tokens
                    )
                elif hasattr(self.model, 'step'):
                    output_tokens = self.model.step(input_tokens)
                else:
                    # Fallback: try __call__
                    output_tokens = self.model(input_tokens)

                # Decode output tokens back to audio
                output_audio = self.mimi.decode(output_tokens)

            elapsed = time.perf_counter() - t
            print(f"[KAME] step: {elapsed*1000:.0f}ms", end="\r")

            # Convert float32 tensor → int16 bytes
            output_np = output_audio.squeeze().cpu().numpy()
            output_int16 = (output_np * 32768).clip(-32768, 32767).astype(np.int16)
            return output_int16.tobytes()

        except Exception as e:
            print(f"[KAME] step error: {e}")
            return b""


# ── Global singleton, loaded once at server startup ──
kame = KameEngine(device="cpu")