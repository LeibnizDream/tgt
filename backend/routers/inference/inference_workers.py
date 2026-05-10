"""
Concrete inference worker implementations for the TGT backend.

This module provides two :class:`~inference.worker.AbstractInferenceWorker`
subclasses that handle different input sources:

- :class:`ZipWorker`      – Processes a locally extracted ZIP archive and
  bundles the output files into a new ZIP for download.
- :class:`OneDriveWorker` – Downloads session folders from a OneDrive share,
  processes them, and uploads the results back to OneDrive.

Both classes are instantiated by the inference router and run in isolated
worker processes so that CPU-intensive ML work does not block the event loop.
"""
import os
import tempfile
import shutil
from zipfile import ZipFile, ZIP_DEFLATED
from pathlib import Path

from inference.abstract_worker import AbstractInferenceWorker
from routers.helpers.onedrive import (
    download_sharepoint_folder,
    upload_file_replace_in_onedrive,
    list_session_children,
)

class ZipWorker(AbstractInferenceWorker):
    """
    Inference worker that processes a locally extracted ZIP archive.

    After the processor runs, the output files listed in
    :attr:`ALLOWED_FILENAMES` are collected from the processed folder tree
    and bundled into a new ZIP archive.  The archive path is reported via the
    job queue so the SSE stream can relay it to the download endpoint.
    """
    # allow-list for files to include in the zip
    ALLOWED_FILENAMES = {
        "trials_and_sessions_annotated.xlsx",
        "transcribed.xlsx",
        "transcription.log",
        "translation.log",
    }

    """
    Processes a single local folder (or multiple if your base_dir contains
    sub-folders named “Session_*”), then zips up only the output files you care
    about and reports the zip path.
    """
    def _initial_message(self):
        """Emit a start-up message before processing begins."""
        self._put("Preparing to process and zip outputs…")
    
    def _folder_to_process(self):
        """Yield the single base directory extracted from the uploaded ZIP."""
        yield self.base_dir

    def _after_process(self):
        """
        After processing each folder, walk the folder tree and zip
        up only the files in ALLOWED_FILENAMES, preserving their
        path relative to self.base_dir.
        """
        base_dir = Path(self.base_dir)
        zip_path = Path(tempfile.gettempdir()) / f"{self.job_id}_output.zip"
        
        try:
            self._put(f"Creating ZIP archive at {zip_path}")
            with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zf:
                for file_path in Path(self.current_folder).rglob("*"):
                    # only include files in our allow-list
                    if file_path.name in self.ALLOWED_FILENAMES:
                        # compute path inside zip relative to the base dir
                        arcname = file_path.relative_to(base_dir)
                        zf.write(file_path, arcname=arcname)
                        print(f"Added to zip: {arcname}")
            print(f"Created ZIP archive at {zip_path}")
            self._put(f"[ZIP PATH] {zip_path}")
        except Exception as e:
            print(f"Failed to create ZIP for job {self.job_id}: {e}")
            self._put(f"[ERROR] Could not bundle outputs: {e}")


class OneDriveWorker(AbstractInferenceWorker):
    """
    Downloads every session folder from a OneDrive share, processes them,
    then uploads the outputs back to OneDrive and cleans up.
    """
    def __init__(self, base_dir, options, token, job):
        super().__init__(base_dir, options, job)
        self.share_link = base_dir
        self.token = token
        self.sessions_meta = []
        self.current_info = {}

    def _initial_message(self):
        self._put("Checking for folders on OneDrive…")
        name_filter = "Session_" if self.options.format == "labvanced" else None
        self.sessions_meta = list_session_children(self.share_link, self.token, name_filter=name_filter)

        if not self.sessions_meta:
            self.sessions_meta = [{"webUrl": self.share_link}]
        self._put(f"Found {len(self.sessions_meta)} folder(s).")

    def _folder_to_process(self):
        for meta in self.sessions_meta:
            if self.cancel.is_set():
                break

            link = meta.get("webUrl")
            self._put("Downloading from OneDrive…")
            self._tempdir_obj = tempfile.TemporaryDirectory(prefix=f"{self.job_id}_")
            self.temp_root = Path(self._tempdir_obj.name)

            try:
                self.current_folder, drive_id, _, sess_map = download_sharepoint_folder(
                    share_link=link,
                    temp_dir=str(self.temp_root),
                    access_token=self.token,
                )
            except Exception as e:
                self._put(f"Failed to download session: {e}. Skipping.")
                shutil.rmtree(self.temp_root, ignore_errors=True)
                continue

            # Determine the session directory on disk
            name = meta.get("name") or next(iter(sess_map.keys()), Path(self.current_folder).name)
            session_path = os.path.join(self.current_folder, name)
            if os.path.isdir(session_path):
                self.current_folder = session_path
            else:
                self._put(f"Warning: expected session folder '{name}' not found; using {self.current_folder}")

            self.current_info = {
                "drive_id": drive_id,
                "sess_map": sess_map,
                "session_name": name,
            }

            yield self.current_folder

    def _after_process(self):
        name = self.current_info["session_name"]
        drive_id = self.current_info["drive_id"]
        sess_map = self.current_info["sess_map"]
        parent_id = sess_map.get(name, "")

        files_to_upload = self._collect_output_files()

        for local_fp in files_to_upload:
            if self.cancel.is_set():
                self._put("[CANCELLED UPLOAD]")
                break
            if not os.path.exists(local_fp):
                continue
            fname = os.path.basename(local_fp)
            self._put(f"Uploading '{fname}' for session '{name}'")
            upload_file_replace_in_onedrive(
                local_file_path=local_fp,
                target_drive_id=drive_id,
                parent_folder_id=parent_id,
                file_name_in_folder=fname,
                access_token=self.token,
            )

        shutil.rmtree(self.temp_root, ignore_errors=True)
        self._put(f"[DONE UPLOADED] {name}")

        self._tempdir_obj.cleanup()
        self._put(f"[INFO] Deleted temporary directory {self.temp_root}")

    def _collect_output_files(self) -> list[str]:
        """Return local paths of all output files that should be uploaded."""
        if self.options.format == "labvanced":
            candidates = [
                "trials_and_sessions_annotated.xlsx",
                f"{self.processor.__class__.__name__}.log",
            ]
            return [
                os.path.join(self.current_folder, f)
                for f in candidates
                if os.path.exists(os.path.join(self.current_folder, f))
            ]

        # plain format: walk and collect all transcribed.xlsx and log files
        results = []
        for root, _, files in os.walk(self.current_folder):
            for f in files:
                if f in ("transcribed.xlsx", f"{self.processor.__class__.__name__}.log"):
                    results.append(os.path.join(root, f))
        return results

