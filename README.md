# ytcc

Long-form audio/video transcription CLI. Transcribes audio in windows cut at pauses, and writes results to `transcript.md` and `segments.jsonl` with global timestamps.

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

Whisper model weights are downloaded automatically on first run (~1.6 GB for the default `large-v3-turbo` model).

## Usage

### Transcribe a file or URL

```bash
# Local file
uv run ytcc transcribe talk.mp4 --out ./output

# YouTube URL (or any yt-dlp supported platform)
uv run ytcc transcribe "https://www.youtube.com/watch?v=XXXX" --out ./output --language zh
```

The output directory must be new or empty. Audio is decoded straight from the source into memory one window at a time, so no intermediate audio is written to disk. Each window is about `--window` seconds long and ends at the quietest point within 30 s of that, so words aren't cut in half. Audio downloaded from a URL is kept until every window succeeded, then deleted.

Options:

| Flag | Default | Description |
|---|---|---|
| `--out` | *(required)* | Output directory (new or empty) |
| `--model` | `large-v3-turbo` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo` |
| `--device` | `auto` | `auto` (CUDA, else Apple Silicon GPU, else CPU), `cpu`, `cuda`, or `mlx` |
| `--compute-type` | `int8` | Quantization: `int8`, `float16`, `float32` (ignored with `mlx`, which runs fp16) |
| `--language` | auto-detect | Language code, e.g. `zh`, `en`, `ja` |
| `--prompt` | | Extra text for Whisper, e.g. topic or names |
| `--glossary` | `~/.config/ytcc/glossary.md` | Glossary file; `--no-glossary` to skip it |
| `--window` | `300` | Target window length in seconds |

The exit status is nonzero when any window failed.

### Meetings

```bash
# One recording
uv run ytcc meeting standup.m4a --out ./standup --summarize

# Microphone and call audio recorded as separate tracks
uv run ytcc meeting --me mic.m4a --others system.m4a --out ./standup --title "Weekly sync"
```

`meeting` takes the same options as `transcribe`, plus `--title`, `--started-at`, `--summarize` and `--summary-model`. With `--me/--others` each track is transcribed separately and the transcript labels lines `Me` or `Others`; lines in the microphone track that repeat the call audio (speakers leaking into the mic) are dropped.

`--summarize`, or `ytcc summarize OUT_DIR` later, writes `summary.md` (overview, topics, decisions, action items, open questions, glossary suggestions) with `claude -p`. This sends the transcript to Anthropic; transcription itself stays local.

### Glossary

`~/.config/ytcc/glossary.md` lists names and terms Whisper should spell correctly, one `- term` bullet each (`- term — note` keeps a note out of the prompt). It is passed to every Whisper decode. Whisper's prompt holds at most 223 tokens (roughly 800 characters of English, 100–200 Chinese characters); terms are kept from the top of the file until that is full, and ytcc warns about the rest, so put the most important terms first. `--summarize` and `ytcc summarize` give Claude the whole glossary, notes included, to correct names in the summary.

### Resume after interruption

```bash
uv run ytcc resume ./output                 # continue where it stopped
uv run ytcc resume ./output --retry-failed  # also redo failed windows
```

`resume` reuses the settings stored in `state.json`; `--device` overrides the device. For meetings it handles every track and rebuilds the merged transcript.

### Clean up

```bash
uv run ytcc clean ./output
```

Removes downloaded audio kept for a resume. Transcript and state are kept.

## Output

```
output/
  transcript.md       # Full transcript with global timestamps
  segments.jsonl      # One JSON record per segment (start, end, text[, speaker])
  state.json          # Settings and progress, for resume
```

A meeting directory also has `meeting.json`, `summary.md` when summarized, and `tracks/<name>/` with each track's own state and transcript; the top-level transcript is the merged one.

`transcript.md` format:

```
[00:00:03] Welcome everyone, let's get started.
[00:00:11] Today we'll be covering the Q3 roadmap.
```

## Architecture

```
ingest/ytdlp.py   URL → local audio file ─┐
audio.py          decode window → numpy   ├→ pipeline → ASR → storage
                                          └→ meeting.py: merge tracks, summarize
```

`pipeline.py` decodes one window at a time from the source, cuts it at a pause, transcribes it, and checkpoints `state.json`. The `ASRBackend` interface is swappable: `asr/faster_whisper_backend.py` serves `cpu`/`cuda`, and `asr/mlx_whisper_backend.py` serves `mlx` (Apple Silicon). `asr/device.py` picks one per `--device`.

## Tests

```bash
uv run pytest
```

## Roadmap

- [ ] Meeting recorder: macOS helper that records microphone + call audio (Core Audio process tap), with a Raycast UI and meeting detection
- [ ] Async streaming pipeline for live streams and live meetings
