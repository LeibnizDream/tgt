"""
Concrete inference worker implementation for the TGT backend.

:class:`OneDriveWorker` downloads session folders from a OneDrive share,
processes them, and uploads the results back to OneDrive.
"""
import os
import tempfile
from pathlib import Path

from inference.abstract_worker import AbstractInferenceWorker
from routers.helpers.onedrive import (
    download_sharepoint_folder,
    list_session_children,
    upload_file_replace_in_onedrive,
)


class OneDriveWorker(AbstractInferenceWorker):
    """
    Downloads every session folder from a OneDrive share, processes them,
    then uploads the outputs back to OneDrive and cleans up.
    """
    def __init__(self, base_dir, options, token, publisher):
        super().__init__(base_dir, options, publisher)
        self.share_link = base_dir
        self.token = token
        self.sessions_meta = []
        self.current_info = {}

    def _initial_message(self):
        self.inform("Checking for folders on OneDrive…")
        name_filter = "Session_" if self.options.format == "labvanced" else None
        self.sessions_meta = list_session_children(self.share_link, self.token, name_filter=name_filter)

        if not self.sessions_meta:
            self.sessions_meta = [{"webUrl": self.share_link}]
        self.inform(f"Found {len(self.sessions_meta)} folder(s).")

    def _folder_to_process(self):
        for meta in self.sessions_meta:
            if self.cancel.is_set():
                break

            link = meta.get("webUrl")
            self.inform("Downloading from OneDrive…")

            with tempfile.TemporaryDirectory() as tmpdir:
                self.temp_root = Path(tmpdir)

                try:
                    self.current_folder, drive_id, _, sess_map = download_sharepoint_folder(
                        share_link=link,
                        temp_dir=str(self.temp_root),
                        access_token=self.token,
                    )
                except Exception as e:
                    self.inform(f"Failed to download session: {e}. Skipping.")
                    continue

                name = meta.get("name") or next(iter(sess_map.keys()), Path(self.current_folder).name)
                session_path = os.path.join(self.current_folder, name)
                if os.path.isdir(session_path):
                    self.current_folder = session_path
                else:
                    self.inform(f"Warning: expected session folder '{name}' not found; using {self.current_folder}")

                self.current_info = {
                    "drive_id": drive_id,
                    "sess_map": sess_map,
                    "session_name": name,
                }

                yield self.current_folder
                # _after_process runs before reaching here; temp_root is cleaned up when with exits

    def _after_process(self):
        name = self.current_info["session_name"]
        drive_id = self.current_info["drive_id"]
        sess_map = self.current_info["sess_map"]
        parent_id = sess_map.get(name, "")

        for local_fp in self._collect_output_files():
            if self.cancel.is_set():
                self.inform("[CANCELLED UPLOAD]")
                break
            if not os.path.exists(local_fp):
                continue
            fname = os.path.basename(local_fp)
            self.inform(f"Uploading '{fname}' for session '{name}'")
            upload_file_replace_in_onedrive(
                local_file_path=local_fp,
                target_drive_id=drive_id,
                parent_folder_id=parent_id,
                file_name_in_folder=fname,
                access_token=self.token,
            )

        self.inform(f"[DONE UPLOADED] {name}")

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

