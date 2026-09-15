# VoxMux Pro &mdash; Multimodal AI Audio/Video Processing Studio

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch CUDA](https://img.shields.io/badge/PyTorch-CUDA%20Accelerated-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end media engineering pipeline that orchestrates deep-learning audio stem separation, synchronized speech-to-text, sequence-to-sequence neural translation, and multi-track container multiplexing.

Engineered with custom runtime hardware profiling to eliminate CUDA Out-Of-Memory (OOM) failures on consumer GPUs.

---

## Key Engineering Features

- **Hardware-Aware Memory Profiling:** Evaluates available GPU VRAM before inference. Dynamically validates quantization constraints (`INT8` vs `FP16`) against model parameter sizes (`base`, `medium`, `large-v3`) to proactively reject overload states.
- **VRAM Cache Management:** Explicit garbage collection and CUDA cache flushing (`torch.cuda.empty_cache()`) executed at stage boundaries to prevent memory accumulation across sub-processes.
- **Dynamic Cross-Lingual Routing:** Automatic source language detection via Faster-Whisper, dynamically mapped to Meta's NLLB FLORES language tokens for translation.
- **Lossless Multiplexing:** Direct stream-copy video muxing via FFmpeg, multiplexing multiple soft subtitle tracks (`.srt`) and isolated audio stems into `.mkv` containers without re-encoding video frames.
- **Asynchronous UI Architecture:** Decoupled execution loop with Python `threading` and `CustomTkinter`, delivering non-blocking real-time terminal stdout redirection to the GUI.

---

## Architectural Pipeline