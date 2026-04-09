"""
Microsoft Graph API helpers for OneDrive / SharePoint operations.

This module provides low-level functions that wrap the Microsoft Graph REST
API to list, download, and upload files in OneDrive / SharePoint document
libraries.  All calls automatically retry with a refreshed OAuth2 token when
a 401 Unauthorized response is received.

Functions:
    :func:`list_session_children`         – List ``Session_*`` child folders
        under a share link.
    :func:`encode_share_link`             – Base64-encode a share link for use
        in Graph API URLs.
    :func:`download_sharepoint_folder`    – Recursively download a SharePoint
        folder to a local directory.
    :func:`upload_file_replace_in_onedrive` – Upload (or replace) a local file
        in a OneDrive drive item.
"""
import os
import requests
import base64

from routers.auth import get_fresh_token


def list_session_children(share_link: str, token: str):
    """
    Helper for the online worker: list all Session_* folders under a OneDrive share link.
    """
    share_id = base64.urlsafe_b64encode(share_link.encode()).decode().rstrip("=")
    url = f"https://graph.microsoft.com/v1.0/shares/u!{share_id}/driveItem/children"
    resp = requests.get(url, headers={"Authorization": f"Bearer {token}"})
    if resp.status_code == 401:
        token = get_fresh_token()
        resp = requests.get(url, headers={"Authorization": f"Bearer {token}"})
    resp.raise_for_status()
    entries = resp.json().get("value", [])
    return [
        entry
        for entry in entries
        if entry.get("folder") and "Session_" in entry["name"]
    ]

def encode_share_link(link):
    """
    Base64url-encode a OneDrive share link for use in Graph API ``shares`` URLs.

    Args:
        link (str): A OneDrive / SharePoint share URL.

    Returns:
        str: Encoded share-link token prefixed with ``u!``.
    """
    encoded_url = base64.urlsafe_b64encode(link.encode()).decode().rstrip("=")
    return f"u!{encoded_url}"

def download_sharepoint_folder(share_link, temp_dir, access_token, file_suffix: list = None):
    """
    Recursively download all files from a SharePoint / OneDrive folder.

    The folder hierarchy is recreated under *temp_dir*. If *file_suffix* is
    provided only files whose names end with one of the given suffixes are
    downloaded (case-insensitive comparison).

    Args:
        share_link (str): OneDrive share URL pointing to the root folder.
        temp_dir (str): Local directory where files will be written.
        access_token (str): OAuth2 bearer token.  Refreshed automatically on
            a 401 response.
        file_suffix (list[str] | None): Whitelist of file extensions to
            download.  ``None`` means download everything.

    Returns:
        tuple: ``(temp_dir, drive_id, parent_folder_id, session_folder_id_map)``
            where *session_folder_id_map* maps each downloaded folder name to
            its Graph API item ID (useful for subsequent uploads).
    """
    share_id = encode_share_link(share_link)
    root_url = f"https://graph.microsoft.com/v1.0/shares/{share_id}/driveItem"

    def get_headers():
        return {"Authorization": f"Bearer {access_token}"}

    def get_with_retry(url):
        nonlocal access_token
        resp = requests.get(url, headers=get_headers())
        if resp.status_code == 401:
            print('code has expired. Renewed automatically')
            access_token = get_fresh_token()
            resp = requests.get(url, headers=get_headers())
        resp.raise_for_status()
        return resp

    root_response = get_with_retry(root_url)
    root_item = root_response.json()

    drive_id = root_item['parentReference']['driveId']
    parent_folder_id = root_item['id']
    session_folder_id_map = {}

    def recursive_collect_files(item, relative_path):
        if "folder" in item:
            folder_path = os.path.join(temp_dir, relative_path, item['name'])
            os.makedirs(folder_path, exist_ok=True)
            session_folder_id_map[item['name']] = item['id']
            children_url = (
                f"https://graph.microsoft.com/v1.0/drives/"
                f"{item['parentReference']['driveId']}/items/{item['id']}/children"
            )
            resp = get_with_retry(children_url)
            for child in resp.json().get('value', []):
                recursive_collect_files(child, os.path.join(relative_path, item['name']))
        else:
            name = item['name']
            if file_suffix is None or any(name.lower().endswith(s.lower()) for s in file_suffix):
                file_folder = os.path.join(temp_dir, relative_path)
                os.makedirs(file_folder, exist_ok=True)
                file_path = os.path.join(file_folder, name)
                download_url = item.get("@microsoft.graph.downloadUrl")
                if download_url:
                    r = requests.get(download_url, stream=True)
                    r.raise_for_status()
                    with open(file_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)

    recursive_collect_files(root_item, relative_path="")
    return temp_dir, drive_id, parent_folder_id, session_folder_id_map


def upload_file_replace_in_onedrive(local_file_path, target_drive_id, parent_folder_id, file_name_in_folder, access_token):
    """
    Upload a local file to OneDrive, replacing any existing file with the same name.

    Uses the Graph API PUT endpoint which creates or replaces the item in a
    single request.  Automatically retries with a refreshed token on a 401
    response and raises descriptive exceptions for 423 (locked), 409
    (conflict), and 403 (forbidden) status codes.

    Args:
        local_file_path (str): Absolute path to the file to upload.
        target_drive_id (str): Graph API drive ID of the destination drive.
        parent_folder_id (str): Graph API item ID of the destination folder.
        file_name_in_folder (str): Name the file will have after upload.
        access_token (str): OAuth2 bearer token.

    Returns:
        dict: The Graph API response JSON for the uploaded item.

    Raises:
        Exception: On HTTP 423 (file locked), 409 (conflict), 403 (forbidden),
            or any other non-2xx response.
    """
    upload_url = f"https://graph.microsoft.com/v1.0/drives/{target_drive_id}/items/{parent_folder_id}:/{file_name_in_folder}:/content"

    def do_upload(token):
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/octet-stream"
        }
        with open(local_file_path, "rb") as f:
            return requests.put(upload_url, headers=headers, data=f)

    resp = do_upload(access_token)
    if resp.status_code == 401:
        print('fresh token has expired, renewing automatically')
        access_token = get_fresh_token()
        resp = do_upload(access_token)

    graph_msg = None
    try:
        j = resp.json()
        graph_msg = j.get("error", {}).get("message")
        graph_code = j.get("error", {}).get("code")
    except Exception:
        graph_code = None

    if resp.status_code == 423:
        raise Exception("Is the file open in another program? Locked (423).")
    elif resp.status_code == 409:
        raise Exception(f"Conflict (409): {graph_code or ''} {graph_msg or resp.text}")
    elif resp.status_code == 403:
        raise Exception(f"Forbidden (403): {graph_code or ''} {graph_msg or resp.text}")
    elif resp.status_code == 401:
        raise Exception("Token refresh failed. Please reconnect to OneDrive.")
    else:
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            raise Exception(f"HTTP {resp.status_code}: {graph_code or ''} {graph_msg or resp.text}") from e

    return resp.json()
