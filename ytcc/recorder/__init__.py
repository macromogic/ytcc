"""
Meeting recording on macOS: `ytcc record`.

A small Swift helper (ytcc_rec.swift, compiled on first use) records the
microphone and the system audio to two files. This module starts and stops it,
keeps the current recording in a state file, and hands finished recordings to
`ytcc meeting` in the background.
"""

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
SOURCES = [HERE / "ytcc_rec.swift", HERE / "Info.plist"]

# Bundle id prefixes of apps whose microphone use most likely means a call.
# Anything else using the microphone is reported too, just without a name.
MEETING_APPS = {
    "us.zoom": "Zoom",
    "com.microsoft.teams": "Microsoft Teams",
    "com.tencent.meeting": "Tencent Meeting",
    "com.tencent.xinWeChat": "WeChat",
    "com.electron.lark": "Feishu",
    "com.bytedance.lark": "Feishu",
    "com.alibaba.DingTalkMac": "DingTalk",
    "com.tinyspeck.slackmacgap": "Slack",
    "com.hnc.Discord": "Discord",
    "com.apple.FaceTime": "FaceTime",
    "Cisco-Systems.Spark": "Webex",
    "com.google.Chrome": "Chrome",
    "com.apple.Safari": "Safari",
    "com.apple.WebKit": "Safari",
    "company.thebrowser": "Arc",
    "org.mozilla.firefox": "Firefox",
    "com.microsoft.edgemac": "Edge",
}
# Microphone users that are never a meeting
IGNORED_APPS = (
    "io.github.macromogic.ytcc-rec",
    "com.apple.SpeechRecognitionCore",
    "com.apple.assistantd",
    "com.finetuneapp.FineTune",  # per-app volume utility; follows other apps' audio
)


class RecorderError(RuntimeError):
    pass


@dataclass
class Recording:
    pid: int
    dir: str
    title: str | None
    started_at: str  # ISO 8601, local time
    summarize: bool


def _state_home() -> Path:
    return Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state") / "ytcc"


def _cache_home() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "ytcc"


def default_meetings_dir() -> Path:
    return Path(os.environ.get("YTCC_MEETINGS_DIR") or Path.home() / "Meetings")


STATE_FILE = _state_home() / "recording.json"


# MARK: helper binary

def helper() -> Path:
    """Path to the compiled helper, building it when its sources changed."""
    if sys.platform != "darwin":
        raise RecorderError("Recording is only supported on macOS.")
    digest = hashlib.sha256(b"".join(p.read_bytes() for p in SOURCES)).hexdigest()[:16]
    binary = _cache_home() / "bin" / f"ytcc-rec-{digest}"
    if binary.exists():
        return binary

    swiftc = shutil.which("swiftc")
    if swiftc is None:
        raise RecorderError("swiftc not found; install the Xcode Command Line Tools (xcode-select --install).")
    binary.parent.mkdir(parents=True, exist_ok=True)
    for old in binary.parent.glob("ytcc-rec-*"):
        old.unlink()
    tmp = binary.with_suffix(".tmp")
    result = subprocess.run(
        [
            swiftc, "-O", str(SOURCES[0]), "-o", str(tmp),
            "-Xlinker", "-sectcreate", "-Xlinker", "__TEXT", "-Xlinker", "__info_plist",
            "-Xlinker", str(SOURCES[1]),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RecorderError(f"Building the recorder failed:\n{result.stderr}")
    tmp.replace(binary)
    return binary


# MARK: state

def current() -> Recording | None:
    if not STATE_FILE.exists():
        return None
    return Recording(**json.loads(STATE_FILE.read_text(encoding="utf-8")))


def _save(recording: Recording) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(asdict(recording), indent=2, ensure_ascii=False), encoding="utf-8")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def status() -> dict:
    rec = current()
    if rec is None:
        return {"recording": False}
    info = asdict(rec)
    info["recording"] = _alive(rec.pid)
    if not info["recording"]:
        info["error"] = "The recorder stopped unexpectedly; `ytcc record stop` saves what was recorded."
    info["elapsed_sec"] = round((datetime.now() - datetime.fromisoformat(rec.started_at)).total_seconds())
    return info


# MARK: detect

def detect() -> dict:
    """Apps using the microphone, and whether one of them looks like a meeting."""
    result = subprocess.run([str(helper()), "detect"], capture_output=True, text=True)
    if result.returncode != 0:
        raise RecorderError(result.stderr.strip())
    rec = current()
    apps = []
    for entry in json.loads(result.stdout)["inputs"]:
        bundle = entry["bundle_id"]
        if bundle.startswith(IGNORED_APPS) or (rec and entry["pid"] == rec.pid):
            continue
        name = next((n for prefix, n in MEETING_APPS.items() if bundle.startswith(prefix)), None)
        apps.append({"bundle_id": bundle, "pid": entry["pid"], "name": name or bundle or f"pid {entry['pid']}"})
    return {"meeting": bool(apps), "apps": apps, "recording": bool(rec and _alive(rec.pid))}


# MARK: start / stop

def _slug(title: str) -> str:
    return re.sub(r"[^\w-]+", "-", title).strip("-")[:60]


def start(title: str | None, base_dir: Path, summarize: bool, echo_cancel: bool) -> Recording:
    rec = current()
    if rec and _alive(rec.pid):
        raise RecorderError(f"Already recording into {rec.dir}.")
    if rec:
        raise RecorderError(f"A previous recording in {rec.dir} was not stopped cleanly; run `ytcc record stop` first.")

    binary = helper()
    now = datetime.now()
    name = now.strftime("%Y-%m-%d_%H%M") + (f"_{_slug(title)}" if title else "")
    out = base_dir.expanduser() / name
    n = 2
    while out.exists():
        out = base_dir.expanduser() / f"{name}-{n}"
        n += 1
    out.mkdir(parents=True)

    log = out / "recorder.log"
    cmd = [str(binary), "record", "--mic", str(out / "me.aac"), "--system", str(out / "others.aac")]
    if echo_cancel:
        cmd.append("--echo-cancel")
    with log.open("w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)

    # Permission prompts on first use can take a while to answer
    deadline = time.monotonic() + 120
    while "recording" not in log.read_text():
        if proc.poll() is not None:
            raise RecorderError(f"The recorder exited:\n{log.read_text().strip()}")
        if time.monotonic() > deadline:
            proc.terminate()
            raise RecorderError("The recorder did not start within 2 minutes.")
        time.sleep(0.2)

    rec = Recording(
        pid=proc.pid, dir=str(out), title=title,
        started_at=now.isoformat(timespec="seconds"), summarize=summarize,
    )
    _save(rec)
    return rec


def stop(transcribe: bool = True) -> Recording:
    """Stop the recorder, finalize its files, and start transcription in the background."""
    rec = current()
    if rec is None:
        raise RecorderError("Not recording.")
    if _alive(rec.pid):
        os.kill(rec.pid, signal.SIGTERM)
        deadline = time.monotonic() + 15
        while _alive(rec.pid):
            if time.monotonic() > deadline:
                os.kill(rec.pid, signal.SIGKILL)  # the .aac files stay readable
                break
            time.sleep(0.1)

    out = Path(rec.dir)
    for track in ("me", "others"):
        _remux(out / f"{track}.aac", out / f"{track}.m4a")
    meta = asdict(rec) | {"ended_at": datetime.now().isoformat(timespec="seconds")}
    del meta["pid"]
    (out / "recording.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    STATE_FILE.unlink(missing_ok=True)

    if transcribe:
        with (out / "transcribe.log").open("w") as f:
            subprocess.Popen(
                [sys.executable, "-m", "ytcc", "record", "process", str(out)],
                stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
            )
    return rec


def _remux(src: Path, dst: Path) -> None:
    """Copy ADTS AAC into MP4, which records the exact duration. Keeps src on failure."""
    if not src.exists():
        return
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-c", "copy", str(dst)],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        src.unlink()


# MARK: post-processing

def process(out: Path) -> int:
    """Transcribe a finished recording with `ytcc meeting`, then notify. Returns the exit status."""
    meta = json.loads((out / "recording.json").read_text(encoding="utf-8"))
    tracks = {t: next((p for p in (out / f"{t}.m4a", out / f"{t}.aac") if p.exists()), None) for t in ("me", "others")}
    if not all(tracks.values()):
        _notify("Recording incomplete", f"Missing audio in {out}")
        return 1

    title = meta.get("title") or datetime.fromisoformat(meta["started_at"]).strftime("Meeting %Y-%m-%d %H:%M")
    cmd = [
        sys.executable, "-m", "ytcc", "meeting",
        "--me", str(tracks["me"]), "--others", str(tracks["others"]),
        "--out", str(out / "transcript"),
        "--title", title,
        "--started-at", meta["started_at"],
    ]
    if meta.get("summarize"):
        cmd.append("--summarize")
    code = subprocess.run(cmd).returncode
    silent = [label for t, label in (("me", "microphone"), ("others", "call audio")) if _silent(tracks[t])]
    if silent:
        # macOS records silence instead of failing when permission is missing
        _notify(
            f"Silent {' and '.join(silent)}",
            "If the meeting had sound there, allow Microphone / System Audio Recording for the app that "
            "started ytcc (Raycast or your terminal) in System Settings > Privacy & Security.",
        )
    elif code == 0:
        _notify("Transcript ready", title)
    if code != 0:
        _notify("Transcription failed", f"{title}: see {out / 'transcribe.log'}")
    return code


def _silent(path: Path) -> bool:
    """Whether a track is digital silence throughout."""
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    match = re.search(r"max_volume: (-?[\d.]+|-inf) dB", result.stderr)
    return bool(match) and (match.group(1) == "-inf" or float(match.group(1)) <= -90)


def _notify(title: str, message: str) -> None:
    script = f"display notification {json.dumps(message)} with title {json.dumps('ytcc: ' + title)}"
    subprocess.run(["osascript", "-e", script], capture_output=True)
