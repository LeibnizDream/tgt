// Hook: job submission (online & upload)

import { useRef } from "react";
import JSZip from "jszip";

type LogType = "info" | "success" | "error" | "warning";

export function useJobSubmission(
  isProcessing: boolean,
  setIsProcessing: (v: boolean) => void,
  addLog: (m: string, t?: LogType) => void,
  streamerOpen: (jobId: string) => void,
) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const submit = async ({
    mode,
    baseDir,
    action,
    instruction,
    language,
    glossingModel,
    translationModel,
    model,
    format,
  }: {
    mode: "online" | "upload";
    baseDir: string;
    action: string;
    instruction: string;
    language: string;
    glossingModel?: string;
    translationModel?: string;
    model?: string;
    format?: string;
  }) => {
    // Hosts where offline uploads are NOT allowed
    const OFFLINE_BLOCKED_HOSTS = new Set<string>([
      "172.20.49.10",
    ]);

    function isOfflineAllowed(hostname: string) {
      return !OFFLINE_BLOCKED_HOSTS.has(hostname);
    }

    const offlineAllowed = typeof window !== "undefined" && isOfflineAllowed(window.location.hostname);

    // Block the upload path on disallowed hosts
    if (mode === "upload" && !offlineAllowed) {
      addLog("Offline upload is disabled on this host.", "warning");
      return;
    }
      

    setIsProcessing(true);
    addLog("Submitting job…", "info");

    const form = new FormData();
    form.append("action", action);
    form.append("instruction", instruction);
    form.append("language", language);
    if (format) {
      form.append("format", format);
    }
    const resolvedModel = model || glossingModel || translationModel;
    if (resolvedModel) {
      form.append("model", resolvedModel);
    }

    if (mode === "online") {
      form.append("base_dir", baseDir);

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
    } else {
      // offline: zip & upload
      addLog("Zipping files…", "info");
      const zip = new JSZip();
      const input = fileInputRef.current!;
      Array.from(input.files || []).forEach((f) => {
        zip.file((f as any).webkitRelativePath, f);
      });
      const blob = await zip.generateAsync({ type: "blob" }, (meta) => {
        addLog(`Zipping ${Math.round(meta.percent)}%`, "info");
      });
      form.append("zipfile", blob, "upload.zip");

      addLog("Uploading zip…", "info");
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/inference/process");
      xhr.withCredentials = true;
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
          addLog(
            `Uploading… ${Math.round((e.loaded / e.total) * 100)}%`,
            "info",
          );
        }
      };
      xhr.onload = () => {
        const { job_id } = JSON.parse(xhr.responseText);
        streamerOpen(job_id);
      };
      xhr.send(form);
    }
  };

  return { fileInputRef, submit };
}
