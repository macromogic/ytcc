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

Whisper model weights are downloaded automatically on first run (~150 MB for the default `base` model).

## Usage

### Transcribe a file

```bash
uv run ytcc transcribe meeting.mp4 --out ./output
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--out` | *(required)* | Output directory |
| `--model` | `base` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large-v3` |
| `--device` | `auto` | `auto` (GPU if available, else CPU), `cpu`, or `cuda` |
| `--compute-type` | `int8` | Quantization: `int8`, `float16`, `float32` |
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
ytcc/
  cli.py                        # click entry point
  config.py                     # default parameters
  models.py                     # Segment dataclass
  ingest/ffmpeg.py              # ffmpeg transcoding + chunking
  asr/base.py                   # ASRBackend interface
  asr/faster_whisper_backend.py # faster-whisper implementation
  chunking/chunk_manager.py     # main loop, global timestamp offset
  storage/state.py              # state.json checkpoint
  storage/transcript_writer.py  # append to .md and .jsonl
  storage/cleanup.py            # delete / move chunks
```

The `ASRBackend` interface is designed to be swappable — a `whisper.cpp` backend can be added without touching the rest of the pipeline.

## Transcribing from a YouTube / livestream URL

Phase 2 is not yet built into the CLI, but the pipeline can be wired up manually with `yt-dlp` and `ffmpeg`:

```bash
# Install yt-dlp (not managed by uv — install globally or in a separate venv)
pip install yt-dlp
```

**Option A — download audio first, then transcribe**

Suitable when disk space is not a concern:

```bash
yt-dlp -f bestaudio -o meeting.%(ext)s "https://www.youtube.com/watch?v=XXXX"
uv run ytcc transcribe meeting.webm --out ./output --language zh
```

**Option B — stream through ffmpeg, avoid storing the full audio**

`yt-dlp` pipes raw audio to `ffmpeg`, which writes fixed-length WAV chunks directly.
A separate process then transcribes and deletes each chunk as it appears.

```bash
# Terminal 1: stream download → ffmpeg chunks
mkdir -p output/chunks
yt-dlp -f bestaudio -o - "https://www.youtube.com/watch?v=XXXX" \
  | ffmpeg -i pipe:0 \
           -vn -ac 1 -ar 16000 \
           -f segment -segment_time 300 \
           output/chunks/chunk_%03d.wav

# Terminal 2: transcribe as chunks arrive
# (run after chunk_000.wav appears; chunk N is complete once chunk N+1 starts)
uv run ytcc transcribe output/chunks/chunk_000.wav --out ./output --language zh
# then resume for subsequent chunks:
uv run ytcc resume ./output
```

> **How chunk completion is detected:** ffmpeg's segment muxer closes `chunk_N.wav` and opens `chunk_N+1.wav` atomically. So `chunk_N` is safe to transcribe as soon as `chunk_N+1` appears on disk. The final chunk is complete when the ffmpeg process exits.

This two-terminal approach will be replaced by an integrated `ytcc transcribe URL` command in Phase 2, which runs both sides as threads within the same process.

## Roadmap

- [ ] Phase 2: YouTube URL input via `yt-dlp` streaming pipeline
- [ ] Overlap + deduplication between chunks
- [ ] LLM-based summarization
