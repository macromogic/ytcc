# ytcc

Long-form audio/video transcription CLI. Segments audio into chunks, transcribes each one, and writes results to `transcript.md` and `segments.jsonl` with global timestamps.

## Requirements

**Python dependencies** are managed by [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

**External dependencies** (install separately, see [DEPENDENCIES.md](DEPENDENCIES.md)):

- `ffmpeg` — audio normalization and chunking
- CUDA + cuDNN — optional, only needed for `--device cuda`

On Apple Silicon Macs, `uv sync` also installs `mlx-whisper`, which runs Whisper on the Metal GPU (`--device mlx`). Other platforms skip it.

`yt-dlp` is included as a Python dependency and installed automatically by `uv sync`.

Whisper model weights are downloaded automatically on first run (~150 MB for the default `base` model).

## Usage

### Transcribe a file or URL

```bash
# Local file
uv run ytcc transcribe meeting.mp4 --out ./output

# YouTube URL (or any yt-dlp supported platform)
uv run ytcc transcribe "https://www.youtube.com/watch?v=XXXX" --out ./output --language zh
```

For URL input, audio is downloaded to a temp file, chunked, transcribed, then cleaned up automatically.

Options:

| Flag | Default | Description |
|---|---|---|
| `--out` | *(required)* | Output directory |
| `--model` | `base` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo` |
| `--device` | `auto` | `auto` (CUDA, else Apple Silicon GPU, else CPU), `cpu`, `cuda`, or `mlx` |
| `--compute-type` | `int8` | Quantization: `int8`, `float16`, `float32` (ignored with `mlx`, which runs fp16) |
| `--language` | auto-detect | Language code, e.g. `zh`, `en`, `ja` |
| `--chunk-duration` | `300` | Chunk length in seconds |

### Resume after interruption

```bash
uv run ytcc resume ./output
```

Already-processed chunks are skipped based on `state.json`.

### Clean leftover chunks

```bash
uv run ytcc clean ./output
```

Removes any unprocessed chunks remaining in `./output/chunks/`. Transcript and state are kept.

## Output

```
output/
  transcript.md       # Full transcript with global timestamps
  segments.jsonl      # One JSON record per segment (start, end, text, chunk)
  state.json          # Checkpoint state for resume
  failed_chunks/      # Chunks that failed transcription (if any)
```

`transcript.md` format:

```
[00:00:03] Welcome everyone, let's get started.
[00:00:11] Today we'll be covering the Q3 roadmap.
```

## Architecture

```
ingest/ytdlp.py   URL → temp audio file ─┐
ingest/ffmpeg.py  file → WAV chunks      ├→ chunk_manager → ASR → storage
```

Both ingest paths converge at the same chunking and transcription pipeline. The `ASRBackend` interface is swappable: `asr/faster_whisper_backend.py` serves `cpu`/`cuda`, and `asr/mlx_whisper_backend.py` serves `mlx` (Apple Silicon). `asr/device.py` picks one per `--device`.

## Roadmap

- [ ] Async streaming pipeline for live streams (yt-dlp | ffmpeg pipe, producer/consumer)
- [ ] Additional platform support (Bilibili, Twitch — yt-dlp handles these already)
- [ ] Overlap + deduplication between chunks
- [ ] LLM-based summarization
