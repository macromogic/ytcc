import { LaunchType, LocalStorage, getPreferenceValues, launchCommand } from "@raycast/api";
import { execFile } from "node:child_process";
import { homedir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

interface Prefs {
  summarize?: boolean;
  notify?: boolean;
  meetingsDir?: string;
  ytccPath?: string;
}

export interface Status {
  recording: boolean;
  dir?: string;
  title?: string | null;
  started_at?: string;
  elapsed_sec?: number;
  error?: string;
}

export interface Detection {
  meeting: boolean;
  recording: boolean;
  apps: { bundle_id: string; pid: number; name: string }[];
}

export const prefs = () => getPreferenceValues<Prefs>();

// Raycast starts commands with a minimal PATH, while ytcc needs ffmpeg,
// swiftc and claude from the user's environment. Ask the login shell once
// and cache the answer.
const ENV_KEY = "shell-env-v1";

interface ShellEnv {
  path: string;
  ytcc: string;
}

async function shellEnv(): Promise<ShellEnv> {
  const cached = await LocalStorage.getItem<string>(ENV_KEY);
  if (cached) return JSON.parse(cached) as ShellEnv;

  const shell = process.env.SHELL || "/bin/zsh";
  const { stdout } = await execFileAsync(
    shell,
    ["-lic", 'printf "\\n__ytcc_env__\\n%s\\n%s\\n" "$PATH" "$(command -v ytcc)"'],
    { timeout: 30_000 },
  );
  const [path = "", ytcc = ""] = (stdout.split("__ytcc_env__\n")[1] ?? "").split("\n");
  const env = { path, ytcc };
  if (env.path) await LocalStorage.setItem(ENV_KEY, JSON.stringify(env));
  return env;
}

/** Run ytcc and return its stdout; errors carry ytcc's own message. */
export async function ytcc(args: string[], timeoutMs = 180_000): Promise<string> {
  for (let attempt = 0; ; attempt++) {
    const env = await shellEnv();
    const command = prefs().ytccPath?.trim() || env.ytcc;
    if (!command) {
      throw new Error("ytcc not found in your login shell; set the “ytcc command” preference.");
    }
    try {
      const { stdout } = await execFileAsync(command, args, {
        env: { ...process.env, PATH: env.path || process.env.PATH },
        timeout: timeoutMs,
      });
      return stdout;
    } catch (error) {
      const err = error as NodeJS.ErrnoException & { stderr?: string };
      if (err.code === "ENOENT" && attempt === 0) {
        await LocalStorage.removeItem(ENV_KEY); // PATH changed; look it up again
        continue;
      }
      const message = err.stderr?.trim().split("\n").pop()?.replace(/^Error: /, "");
      throw new Error(message || err.message);
    }
  }
}

export const status = async () => JSON.parse(await ytcc(["record", "status"])) as Status;
export const detect = async () => JSON.parse(await ytcc(["record", "detect"])) as Detection;

export async function startRecording(title?: string): Promise<void> {
  const p = prefs();
  const args = ["record", "start"];
  if (title?.trim()) args.push("--title", title.trim());
  if (p.summarize) args.push("--summarize");
  if (p.meetingsDir?.trim()) args.push("--dir", expandHome(p.meetingsDir.trim()));
  await ytcc(args);
}

export async function stopRecording(transcribe = true): Promise<void> {
  await ytcc(transcribe ? ["record", "stop"] : ["record", "stop", "--no-transcribe"]);
}

export function meetingsDir(): string {
  const dir = prefs().meetingsDir?.trim();
  return dir ? expandHome(dir) : join(homedir(), "Meetings");
}

function expandHome(path: string): string {
  return path.startsWith("~/") ? join(homedir(), path.slice(2)) : path;
}

export async function notify(title: string, message: string): Promise<void> {
  const script = `display notification ${JSON.stringify(message)} with title ${JSON.stringify(title)}`;
  await execFileAsync("osascript", ["-e", script]).catch(() => undefined);
}

/** Refresh the menu bar item after a change made elsewhere. */
export async function refreshMenuBar(): Promise<void> {
  await launchCommand({ name: "menu-bar", type: LaunchType.Background }).catch(() => undefined);
}

export const errorMessage = (error: unknown) => (error instanceof Error ? error.message : String(error));
