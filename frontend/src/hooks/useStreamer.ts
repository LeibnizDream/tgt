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
    const data = e.data;
    if (data === "[PING]") return;

    if (data.startsWith("[PROGRESS]")) {
      const [cur, tot] = data.replace("[PROGRESS]", "").trim().split("/").map(Number);
      setProgress?.(cur, tot);
      return;
    }

    if (data.includes("[ERROR]")) {
      addLog(data, "error");
      return finish();
    }

    if (data.includes("[WARNING]")) {
      addLog(data, "warning");
      return;
    }

    if (data === "[DONE ALL]") {
      doneRef.current = true;
      addLog("Workflow completed successfully!", "success");
      if (prefix === "inference") {
        const downloadUrl = `/api/${prefix}/${jobId}/download`;
        try {
          // try to GET the zip; if it 404s, this'll go to catch
          const res = await fetch(downloadUrl);
          if (!res.ok) throw new Error(`No ZIP (status ${res.status})`);

          // pull it down as a blob…
          const blob = await res.blob();
          const blobUrl = window.URL.createObjectURL(blob);

          // …and trigger the download
          const a = document.createElement("a");
          a.href = blobUrl;
          a.download = `${jobId}_results.zip`;
          document.body.appendChild(a);
          a.click();
          a.remove();
          window.URL.revokeObjectURL(blobUrl);

          addLog("Download started…", "info");
        } catch (err) {
          // either a 404 or network error → no files to download
          addLog("No files to download.");
        } finally {
          finish();
        }
      } else {
        addLog("Model saved in models!", "success");
        finish();
      }
    } else {
      addLog(data, "info");
    }
  };

  }; // ← **this** closes the open() function

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
