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

### Whisper model weights
Downloaded automatically by `faster-whisper` on first use to `~/.cache/huggingface/hub/`.
Default model is `base` (~150 MB). Available sizes: `tiny`, `base`, `small`, `medium`, `large-v3`.
