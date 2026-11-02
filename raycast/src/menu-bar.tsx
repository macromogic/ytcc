import { Color, Icon, LocalStorage, MenuBarExtra, open, showHUD } from "@raycast/api";
import { useEffect, useState } from "react";
import {
  Detection,
  Status,
  detect,
  errorMessage,
  meetingsDir,
  notify,
  prefs,
  startRecording,
  status,
  stopRecording,
} from "./ytcc";

interface State {
  status?: Status;
  detection?: Detection;
  error?: string;
}

const SEEN_KEY = "seen-mic-pids";

async function load(): Promise<State> {
  try {
    const s = await status();
    if (s.recording || s.dir) return { status: s };
    const detection = await detect();
    await notifyNewApps(detection);
    return { status: s, detection };
  } catch (error) {
    return { error: errorMessage(error) };
  }
}

/** Notify once per app process that starts using the microphone. */
async function notifyNewApps(detection: Detection): Promise<void> {
  const seen = new Set<number>(JSON.parse((await LocalStorage.getItem<string>(SEEN_KEY)) ?? "[]"));
  const fresh = detection.apps.filter((app) => !seen.has(app.pid));
  await LocalStorage.setItem(SEEN_KEY, JSON.stringify(detection.apps.map((app) => app.pid)));
  if (fresh.length > 0 && prefs().notify !== false) {
    const names = fresh.map((app) => app.name).join(", ");
    await notify("Meeting detected?", `${names} is using the microphone. Record it from the menu bar.`);
  }
}

export default function Command() {
  const [state, setState] = useState<State>();
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    load().then(setState);
  }, []);

  async function act(action: () => Promise<void>, done: string) {
    setBusy(true);
    try {
      await action();
      await showHUD(done);
    } catch (error) {
      await showHUD(`ytcc: ${errorMessage(error)}`);
    }
    setState(await load());
    setBusy(false);
  }

  const s = state?.status;
  const apps = state?.detection?.apps ?? [];
  const isLoading = state === undefined || busy;

  const start = (
    <MenuBarExtra.Item
      title="Start Recording"
      icon={Icon.CircleFilled}
      onAction={() => act(() => startRecording(), "● Recording meeting")}
    />
  );
  const footer = (
    <MenuBarExtra.Section>
      <MenuBarExtra.Item title="Open Recordings Folder" icon={Icon.Folder} onAction={() => open(meetingsDir())} />
    </MenuBarExtra.Section>
  );

  if (state?.error) {
    return (
      <MenuBarExtra icon={Icon.Warning} tooltip={`ytcc: ${state.error}`} isLoading={isLoading}>
        <MenuBarExtra.Item title={state.error} />
        {footer}
      </MenuBarExtra>
    );
  }

  if (s?.recording) {
    const minutes = Math.floor((s.elapsed_sec ?? 0) / 60);
    return (
      <MenuBarExtra
        icon={{ source: Icon.CircleFilled, tintColor: Color.Red }}
        title={`${minutes} min`}
        tooltip={`Recording ${s.title ?? "meeting"}`}
        isLoading={isLoading}
      >
        <MenuBarExtra.Section title={`Recording ${s.title ?? "meeting"} · ${minutes} min`}>
          <MenuBarExtra.Item
            title="Stop and Transcribe"
            icon={Icon.Stop}
            onAction={() => act(() => stopRecording(), "Saved; transcribing in the background")}
          />
          <MenuBarExtra.Item
            title="Stop without Transcribing"
            icon={Icon.Stop}
            onAction={() => act(() => stopRecording(false), "Recording saved")}
          />
          <MenuBarExtra.Item title="Open Recording Folder" icon={Icon.Folder} onAction={() => open(s.dir!)} />
        </MenuBarExtra.Section>
        {footer}
      </MenuBarExtra>
    );
  }

  if (s?.dir) {
    // State left behind by a recorder that died
    return (
      <MenuBarExtra icon={{ source: Icon.Warning, tintColor: Color.Orange }} isLoading={isLoading}>
        <MenuBarExtra.Section title={s.error ?? "The recorder stopped unexpectedly"}>
          <MenuBarExtra.Item
            title="Save and Transcribe"
            icon={Icon.Stop}
            onAction={() => act(() => stopRecording(), "Saved; transcribing in the background")}
          />
        </MenuBarExtra.Section>
        {footer}
      </MenuBarExtra>
    );
  }

  if (apps.length > 0) {
    return (
      <MenuBarExtra
        icon={{ source: Icon.Microphone, tintColor: Color.Red }}
        title="Record?"
        tooltip="An app is using the microphone"
        isLoading={isLoading}
      >
        <MenuBarExtra.Section title={`Using the microphone: ${apps.map((app) => app.name).join(", ")}`}>
          {start}
        </MenuBarExtra.Section>
        {footer}
      </MenuBarExtra>
    );
  }

  return (
    <MenuBarExtra icon={Icon.Microphone} tooltip="ytcc meeting recorder" isLoading={isLoading}>
      {start}
      {footer}
    </MenuBarExtra>
  );
}
