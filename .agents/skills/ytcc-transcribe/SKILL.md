---
name: ytcc-transcribe
description: Transcribe a local audio or video file, or a yt-dlp-supported video URL, with this repository's ytcc CLI. Optionally summarize the resulting transcript when the user asks for a summary or notes.
---

# Transcribe with ytcc

Run the `ytcc` command from any directory; machines set up by nix-config have it on PATH. Without it, use `uv run ytcc` from the ytcc repository root instead. The CLI turns a local media file or supported URL into `transcript.md`, `segments.jsonl`, and `state.json`. Summarization is a separate, optional step performed from those outputs.

## Run

1. Identify the source and choose a new, dedicated output directory. Never run `ytcc transcribe` into a directory containing an earlier transcript: the CLI appends to its output files.
2. Run `ytcc transcribe SOURCE --out OUT_DIR`. Quote the source and output path. Pass user-requested CLI options such as `--model`, `--device`, or `--language`. For Chinese/English mixed speech, leave language auto-detection on unless the user specifies otherwise; mention that the current backend may still make mistakes at language switches.
3. Inspect `state.json` after the command. Consider the transcript complete only when `failed_chunks` is empty and the number of `processed_chunks` equals `total_chunks`. Report failures or missing output clearly. Do not treat the CLI's zero exit status or “Done” message alone as proof of completeness.
4. Return the paths to `transcript.md` and `segments.jsonl`, plus any material limits noticed during the run. Keep the original media file.

URLs must be supported by yt-dlp and accessible to the user; site restrictions or authentication can still block download. Do not work around access controls. First use may download Whisper model weights.

## Optional summary

Summarize only when requested. Read the entire completed transcript, using `segments.jsonl` for timestamped evidence. For a long recording, process it in time order and cover all chunks before combining notes. Write `summary.md` in the same output directory unless the user requests another location.

Choose a structure that fits the content: for a meeting, capture topics, decisions, action items, owners, and open questions; for a talk or video, capture the main ideas. Include timestamps for substantive claims. Preserve the source language or follow the user's requested output language. Mark uncertain names, owners, and decisions as uncertain, and avoid inventing speaker identities; current ytcc output has no speaker labels. If transcription is partial, explain the gap and only summarize it when the user explicitly requests a partial summary.

## Resume

`ytcc resume OUT_DIR` can continue remaining chunks, but the current state file does not preserve model or language settings. Reuse the same options from the original run. Failed chunks are moved to `failed_chunks/` and are not retried by `resume` automatically. Do not rerun `transcribe` in the same output directory to recover from a failure.
