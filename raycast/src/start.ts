import { LaunchProps, showHUD } from "@raycast/api";
import { errorMessage, refreshMenuBar, startRecording } from "./ytcc";

export default async function Command(props: LaunchProps<{ arguments: { title?: string } }>) {
  try {
    await startRecording(props.arguments.title);
    await showHUD("● Recording meeting");
  } catch (error) {
    await showHUD(`ytcc: ${errorMessage(error)}`);
  }
  await refreshMenuBar();
}
