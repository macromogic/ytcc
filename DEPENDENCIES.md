# External Dependencies

These are not managed by uv and must be installed separately.

## Required

### ffmpeg
Used for audio normalization and chunking.

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Windows (via winget)
winget install ffmpeg
```

## Optional

### CUDA + cuDNN (for GPU acceleration)
`faster-whisper` can use NVIDIA GPUs. Required only if passing `--device cuda`.

- Install the CUDA toolkit matching your driver: https://developer.nvidia.com/cuda-downloads
- cuDNN 8.x is required by CTranslate2 (the faster-whisper backend)

### Apple Silicon GPU (MLX)
Nothing extra to install: `mlx-whisper` is a Python dependency limited to macOS arm64, so `uv sync` installs it on Apple Silicon only. Used by `--device mlx` (and by `--device auto` on those Macs).

### Whisper model weights
Downloaded automatically on first use to `~/.cache/huggingface/hub/`.
Default model is `large-v3-turbo` (~1.6 GB). Available sizes: `tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo`.
The `mlx` device uses separate MLX-format weights from the `mlx-community` Hugging Face org, so the first run on a new device downloads that model again.
