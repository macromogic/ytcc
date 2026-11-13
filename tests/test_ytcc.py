import shutil
import subprocess

import numpy as np
import pytest
from click.testing import CliRunner

from ytcc import audio, glossary, pipeline
from ytcc.asr.base import ASRBackend
from ytcc.cli import cli
from ytcc.meeting import drop_echo
from ytcc.models import Segment
from ytcc.storage.state import Options, State, load_state
from ytcc.storage.transcript_writer import read_segments

SR = audio.SAMPLE_RATE


class FakeBackend(ASRBackend):
    """Returns one segment per window; fails on the calls listed in fail_calls."""

    def __init__(self, fail_calls=()):
        self.calls = 0
        self.fail_calls = set(fail_calls)

    def transcribe(self, samples, language):
        self.calls += 1
        if self.calls in self.fail_calls:
            raise RuntimeError("boom")
        return [
            Segment(0.5, len(samples) / SR, f"call {self.calls}"),
            Segment(0.5, 3, "Thank you."),  # over silence, see fake_speech
        ]

    def detect_language(self, samples):
        return "en", 1.0

    def count_tokens(self, text):
        return len(text)


def fake_decode(source, start, length, real_duration=200):
    # 1 kHz tone with a silent second every 10 s (at 9-10 s, 19-20 s, ...)
    stop = real_duration if length is None else min(start + length, real_duration)
    t = np.arange(int(start * SR), int(stop * SR)) / SR
    tone = 0.3 * np.sin(2 * np.pi * 1000 * t)
    tone[(t % 10) >= 9] = 0
    return tone.astype(np.float32)


def fake_speech(samples):
    # Speech everywhere except the first 4 s
    return [(4.0, len(samples) / SR)]


def run(out_dir, state, backend, **kwargs):
    pipeline.run(out_dir, state, backend, decode=fake_decode, speech=fake_speech, **kwargs)


def make_state(duration, window=60):
    options = Options("tiny", "cpu", "int8", None, None, window)
    return State(source="x.wav", audio="x.wav", duration=duration, options=options)


def test_find_cut_picks_silence():
    samples = fake_decode(None, 0, 80)
    cut = audio.find_cut(samples, target=60, search=5)
    assert 59 <= cut <= 60


def test_windows_cut_in_pauses_and_cover_everything(tmp_path):
    state = make_state(duration=200)
    run(tmp_path, state, FakeBackend())

    assert state.complete
    assert state.windows[0].start == 0
    assert state.windows[-1].end == 200
    for prev, nxt in zip(state.windows, state.windows[1:]):
        assert prev.end == nxt.start
        assert 9 <= prev.end % 10 <= 10  # cut inside a pause
    # Segment times are global; "Thank you." over silence is dropped
    segments = read_segments(tmp_path)
    assert [round(s.start - 0.5, 3) for s in segments] == [round(w.start, 3) for w in state.windows]
    assert (tmp_path / "transcript.md").read_text().startswith("[00:00:00] call 1")


@pytest.mark.parametrize("probed", [150, 260])
def test_probed_duration_off(tmp_path, probed):
    # Real audio is 200 s; the probed duration is only an estimate
    state = make_state(duration=probed)
    run(tmp_path, state, FakeBackend())
    assert state.complete
    assert state.windows[-1].end == pytest.approx(200, abs=0.01)
    assert state.duration == pytest.approx(200, abs=0.01)


def test_failed_window_is_retried(tmp_path):
    state = make_state(duration=200)
    run(tmp_path, state, FakeBackend(fail_calls={2}))
    assert len(state.failed) == 1 and not state.complete

    state = load_state(tmp_path)  # resume from disk
    run(tmp_path, state, FakeBackend(), retry_failed=True)
    assert state.complete
    starts = [s.start for s in read_segments(tmp_path)]
    assert starts == sorted(starts) and len(starts) == len(state.windows)


def test_glossary(tmp_path):
    path = tmp_path / "glossary.md"
    path.write_text("# People\n- Whexy — colleague\n- 张三\n\nnot a term\n- Whexy\n", encoding="utf-8")
    terms = glossary.load_terms(path)
    assert terms == ["Whexy", "张三"]
    assert glossary.build_prompt(terms, "weekly sync") == "weekly sync; Whexy, 张三"
    assert glossary.build_prompt([], None) is None


def test_fit_prompt_keeps_terms_in_order():
    terms = ["alpha", "beta", "gamma"]
    prompt, dropped = glossary.fit_prompt(terms, "x", len, limit=len("x; alpha, beta"))
    assert prompt == "x; alpha, beta" and dropped == ["gamma"]
    with pytest.raises(ValueError):
        glossary.fit_prompt(terms, "too long", len, limit=3)


def test_drop_echo():
    others = [Segment(10, 14, "Let's ship the release on Friday.")]
    mine = [
        Segment(10.5, 14, "let's ship the release on friday"),  # echo from speakers
        Segment(12, 13, "Sounds good to me."),
        Segment(30, 32, "Let's ship the release on Friday."),  # repeated later: real speech
    ]
    assert [s.text for s in drop_echo(mine, others)] == ["Sounds good to me.", "Let's ship the release on Friday."]


def test_transcribe_refuses_nonempty_out(tmp_path):
    (tmp_path / "transcript.md").write_text("old")
    result = CliRunner().invoke(cli, ["transcribe", "x.wav", "--out", str(tmp_path)])
    assert result.exit_code != 0
    assert "not empty" in result.output


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_decode_with_ffmpeg(tmp_path):
    path = tmp_path / "tone.m4a"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=5", str(path)],
        check=True,
    )
    assert abs(audio.probe_duration(path) - 5) < 0.1
    samples = audio.decode(path, 1, 2)
    assert abs(len(samples) / SR - 2) < 0.05
    assert samples.dtype == np.float32 and np.abs(samples).max() > 0.1


def test_resume_after_stop_between_segments_and_state(tmp_path):
    # A run stopped after a window's segments were written but before state.json
    # recorded the window: resume transcribes it again without duplicating it
    state = make_state(duration=200)
    run(tmp_path, state, FakeBackend())
    expected = [(s.start, s.end) for s in read_segments(tmp_path)]

    state = load_state(tmp_path)
    last = state.windows.pop()
    state.position = last.start
    run(tmp_path, state, FakeBackend())

    assert state.complete
    assert [(s.start, s.end) for s in read_segments(tmp_path)] == expected


def test_faster_whisper_does_not_condition_on_previous_text(monkeypatch):
    from ytcc.asr import faster_whisper_backend as fw

    seen = {}

    class FakeModel:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, audio, **kwargs):
            seen.update(kwargs)
            return [], None

    monkeypatch.setattr(fw, "WhisperModel", FakeModel)
    fw.FasterWhisperBackend(prompt="Kubernetes").transcribe(np.zeros(SR, np.float32), "en")
    assert seen["condition_on_previous_text"] is False
    assert seen["hotwords"] == "Kubernetes"


def test_failed_decodes_and_low_confidence_loops_are_dropped():
    def seg(t, text, logprob=-0.3, ratio=1.2):
        return Segment(t, t + 0.5, text, avg_logprob=logprob, compression_ratio=ratio)

    everywhere = [(0.0, 100.0)]
    # Whisper gave up on the decode: its text compresses too well
    assert not pipeline._is_speech(seg(1, "Just...", ratio=15.9), everywhere)
    assert pipeline._is_speech(seg(1, "Just...", ratio=2.0), everywhere)

    loop = [seg(t, "Am I", logprob=-2.67) for t in range(10, 19)]
    backchannels = [seg(t, "嗯", logprob=-1.47) for t in range(20, 25)]
    short_run = [seg(30, "skunk", logprob=-3.9), seg(31, "skunk", logprob=-3.9)]
    kept = pipeline._drop_loops([seg(0, "Hello"), *loop, *backchannels, *short_run])
    assert [s.text for s in kept] == ["Hello"] + ["嗯"] * 5 + ["skunk"] * 2

    # Backends that report no confidence keep everything
    unknown = [Segment(t, t + 0.5, "Am I") for t in range(5)]
    assert pipeline._drop_loops(unknown) == unknown
