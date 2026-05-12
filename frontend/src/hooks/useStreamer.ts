import { useEffect, useRef } from "react";

type LogType = "info" | "success" | "error" | "warning";

const JOB_KEY = "job_id";

export function useStreamer(
  addLog: (msg: string, type?: LogType) => void,
  setIsProcessing: (v: boolean) => void,
  prefix: "inference" | "train",
  setProgress?: (current: number, total: number) => void
) {
  const evtRef = useRef<EventSource | null>(null);
  const doneRef = useRef(false);

  const finish = () => {
    doneRef.current = false;
    evtRef.current?.close();
    evtRef.current = null;
    setIsProcessing(false);
    setProgress?.(0, 0);
    localStorage.removeItem(JOB_KEY);
  };

  const open = (jobId: string) => {
    localStorage.setItem(JOB_KEY, jobId);
    addLog(`Opened job ${jobId}`, "info");
    setIsProcessing(true);
    setProgress?.(0, 0);

    const evt = new EventSource(`/api/${prefix}/${jobId}/stream`);
    evtRef.current = evt;

    evt.onerror = () => {
      if (!doneRef.current) {
        addLog("Connection to server lost. The job may still be running.", "error");
        finish();
      }
    };

    evt.onmessage = async (e) => {
      let msg: { type: string; message?: string; current?: number; total?: number };
      try {
        msg = JSON.parse(e.data);
      } catch {
        addLog(e.data, "info");
        return;
      }

      if (msg.type === "ping") return;

      if (msg.type === "progress") {
        setProgress?.(msg.current ?? 0, msg.total ?? 0);
        return;
      }

      if (msg.type === "error") {
        addLog(msg.message ?? "Unknown error", "error");
        return finish();
      }

      if (msg.type === "cancelled") {
        addLog("Job cancelled.", "warning");
        return finish();
      }

      if (msg.type === "done") {
        doneRef.current = true;
        addLog("Workflow completed successfully!", "success");
        if (prefix === "train") {
          addLog("Model saved in models!", "success");
        }
        finish();
        return;
      }

      if (msg.type === "warning") {
        addLog(msg.message ?? "", "warning");
        return;
      }

      addLog(msg.message ?? "", "info");
    };
  };

  const cancel = () => {
    const jobId = localStorage.getItem(JOB_KEY);
    if (!jobId) return;
    fetch(`/api/${prefix}/cancel`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_id: jobId }),
      credentials: "same-origin",
    }).then(() => {
      addLog("Cancelled", "warning");
      finish();
    });
  };

  useEffect(() => {
    const pending = localStorage.getItem(JOB_KEY);
    if (pending) open(pending);
    return () => evtRef.current?.close();
  }, []);

  return { open, cancel };
}
