import { showHUD } from "@raycast/api";
import { errorMessage, refreshMenuBar, stopRecording } from "./ytcc";

export default async function Command() {
  try {
    await stopRecording();
    await showHUD("Recording saved; transcribing in the background");
  } catch (error) {
    await showHUD(`ytcc: ${errorMessage(error)}`);
  }
  await refreshMenuBar();
}
