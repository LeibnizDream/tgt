type LogType = "info" | "success" | "error" | "warning";

export function useTrainSubmission(
  _isProcessing: boolean,
  setIsProcessing: (v: boolean) => void,
  addLog: (m: string, t?: LogType) => void,
  streamerOpen: (jobId: string) => void,
) {
  const submit = async ({
    baseDir,
    action,
    study,
    language,
  }: {
    baseDir: string;
    action: string;
    study: string;
    language: string;
  }) => {
    setIsProcessing(true);
    addLog("Submitting job…", "info");

    const form = new FormData();
    form.append("action", action);
    form.append("language", language);
    form.append("study", study);
    form.append("base_dir", baseDir);

    try {
      const res = await fetch("/api/train/process", {
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
      setIsProcessing(false);
      streamerOpen(job_id);
    } catch (err) {
      addLog(`Submission failed: ${err}`, "error");
      setIsProcessing(false);
    }
  };

  return { submit };
}
