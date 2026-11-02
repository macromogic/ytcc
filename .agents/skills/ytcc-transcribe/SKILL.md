---
name: ytcc-transcribe
description: Transcribe a local audio or video file, a meeting recording, or a yt-dlp-supported video URL with this repository's ytcc CLI. Optionally summarize the resulting transcript.
---

# Transcribe with ytcc

Run the `ytcc` command from any directory; machines set up by nix-config have it on PATH. Without it, use `uv run ytcc` from the ytcc repository root instead. Transcription runs locally. Summaries via `--summarize` or `ytcc summarize` send the transcript to Anthropic through `claude -p`.

## Run

1. Identify the source and choose a new output directory. ytcc refuses a non-empty one.
2. For a meeting, run `ytcc meeting SOURCE --out OUT_DIR`, or `ytcc meeting --me MIC_TRACK --others SYSTEM_TRACK --out OUT_DIR` when the recording has the microphone and the call audio as separate files. Add `--title` and `--summarize` when wanted. For anything else, run `ytcc transcribe SOURCE --out OUT_DIR`. Quote paths. Pass user-requested options such as `--model`, `--device` or `--language`; leave language auto-detection on for Chinese/English mixed speech unless told otherwise.
3. A nonzero exit status means the transcript is incomplete. Check `state.json` (in each `tracks/*/` for meetings): `windows` marked `failed` can be retried with `ytcc resume OUT_DIR --retry-failed`; a `position` short of `duration` means the run was interrupted, and `ytcc resume OUT_DIR` continues it with the original settings.
4. Return the paths to `transcript.md`, `segments.jsonl` and `summary.md` if written. Keep the original media files.

URLs must be supported by yt-dlp and accessible to the user; do not work around access controls. First use may download Whisper model weights.

## Recordings

`ytcc record start/stop` (macOS) saves meetings under `~/Meetings/` (or `$YTCC_MEETINGS_DIR`) and runs `ytcc meeting` on them automatically; the transcript is in each recording's `transcript/` directory. To redo one, run `ytcc meeting --me me.m4a --others others.m4a --out NEW_DIR` from its folder.

## Speakers

With `--me/--others`, lines are labeled `Me` (the user's microphone) and `Others` (everyone else in the call, mixed). There is no further speaker separation; never invent identities for "Others".

## Summary

Summarize only when requested. Prefer `ytcc summarize OUT_DIR` (or `--summarize`), which writes `summary.md` with overview, topics, decisions, action items, open questions and glossary suggestions. When summarizing by hand instead, read the whole transcript, use timestamps for claims, follow the transcript's language, mark uncertain owners and decisions as uncertain, and only summarize a partial transcript when the user explicitly asks.

## Glossary

`~/.config/ytcc/glossary.md` holds names and terms Whisper should spell correctly: one `- term` bullet per term, optionally `- term — note` (the note is not sent to Whisper). It is used automatically; `--no-glossary` disables it. Whisper's prompt holds 223 tokens (roughly 800 English characters or 100–200 Chinese characters); terms are kept from the top until it is full, and ytcc warns about the rest. Summaries get the whole glossary.
