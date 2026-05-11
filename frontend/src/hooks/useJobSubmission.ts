type LogType = "info" | "success" | "error" | "warning";

export function useJobSubmission(
  _isProcessing: boolean,
  setIsProcessing: (v: boolean) => void,
  addLog: (m: string, t?: LogType) => void,
  streamerOpen: (jobId: string) => void,
) {
  const submit = async ({
    baseDir,
    action,
    instruction,
    language,
    model,
    format,
  }: {
    baseDir: string;
    action: string;
    instruction: string;
    language: string;
    model?: string;
    format?: string;
  }) => {
    setIsProcessing(true);
    addLog("Submitting job…", "info");

    const form = new FormData();
    form.append("action", action);
    form.append("instruction", instruction);
    form.append("language", language);
    form.append("base_dir", baseDir);
    if (format) form.append("format", format);
    if (model) form.append("model", model);

    const res = await fetch("/api/inference/process", {
      method: "POST",
      body: form,
      credentials: "same-origin",
    });

    if (!res.ok) {
      if (res.status === 401) {
        addLog("OneDrive session expired. Please reconnect.", "error");
      } else {
        const errorText = await res.text();
        addLog(`Error: ${errorText}`, "error");
      }
      setIsProcessing(false);
      return;
    }

    const { job_id } = await res.json();
    streamerOpen(job_id);
  };

  return { submit };
}
