"""Google Drive access through a service account.

The recruiter shares one or more folders with the service-account e-mail; this
module lists everything the service account can see and downloads the CV files
inside a chosen folder.  ``DriveClient`` is a Protocol so tests (and mock mode)
can supply an in-memory stand-in.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

logger = logging.getLogger(__name__)

FOLDER_MIME = "application/vnd.google-apps.folder"
PDF_MIME = "application/pdf"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
SUPPORTED_MIMES = (PDF_MIME, DOCX_MIME)
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


@dataclass
class DriveFolderInfo:
    id: str
    name: str

    def as_dict(self) -> dict[str, str]:
        return {"id": self.id, "name": self.name}


@dataclass
class DriveFileInfo:
    id: str
    name: str
    mime_type: str
    md5_checksum: str | None
    size: int
    folder_id: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "mime_type": self.mime_type,
            "md5_checksum": self.md5_checksum,
            "size": self.size,
            "folder_id": self.folder_id,
        }


class DriveClient(Protocol):
    def status(self) -> dict[str, Any]: ...

    def list_folders(self) -> list[DriveFolderInfo]: ...

    def list_files(self, folder_id: str) -> list[DriveFileInfo]: ...

    def download(self, file_id: str) -> bytes: ...


class DriveAuthError(RuntimeError):
    """The service account could not be loaded or authorised."""


# ---------------------------------------------------------------------------
# Real Drive client
# ---------------------------------------------------------------------------


class GoogleDriveClient:
    """Service-account backed Drive client (the only Drive-SDK importer)."""

    def __init__(self, service_account_json: str, *, page_size: int = 200, service: Any = None):
        self.service_account_json = service_account_json
        self.page_size = page_size
        self._service = service
        self._email = ""

    def _build(self):
        if self._service is not None:
            return self._service
        path = Path(self.service_account_json)
        if not path.is_file():
            raise DriveAuthError(
                f"service account JSON not found at {self.service_account_json!r}; "
                "set GOOGLE_SERVICE_ACCOUNT_JSON"
            )
        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise DriveAuthError("google-api-python-client is not installed") from exc

        info = json.loads(path.read_text(encoding="utf-8"))
        self._email = info.get("client_email", "")
        creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
        self._service = build("drive", "v3", credentials=creds, cache_discovery=False)
        return self._service

    def status(self) -> dict[str, Any]:
        try:
            self._build()
        except DriveAuthError as exc:
            return {"connected": False, "service_account_email": self._email, "error": str(exc)}
        return {"connected": True, "service_account_email": self._email, "error": None}

    def list_folders(self) -> list[DriveFolderInfo]:
        service = self._build()
        query = f"mimeType='{FOLDER_MIME}' and trashed=false"
        folders: list[DriveFolderInfo] = []
        token = None
        while True:
            response = (
                service.files()
                .list(
                    q=query,
                    spaces="drive",
                    fields="nextPageToken, files(id, name)",
                    pageSize=self.page_size,
                    pageToken=token,
                    includeItemsFromAllDrives=True,
                    supportsAllDrives=True,
                )
                .execute()
            )
            folders.extend(DriveFolderInfo(f["id"], f.get("name", "")) for f in response.get("files", []))
            token = response.get("nextPageToken")
            if not token:
                break
        return folders

    def list_files(self, folder_id: str) -> list[DriveFileInfo]:
        service = self._build()
        mimes = " or ".join(f"mimeType='{m}'" for m in SUPPORTED_MIMES)
        query = f"'{folder_id}' in parents and trashed=false and ({mimes})"
        files: list[DriveFileInfo] = []
        token = None
        while True:
            response = (
                service.files()
                .list(
                    q=query,
                    spaces="drive",
                    fields="nextPageToken, files(id, name, mimeType, md5Checksum, size)",
                    pageSize=self.page_size,
                    pageToken=token,
                    includeItemsFromAllDrives=True,
                    supportsAllDrives=True,
                )
                .execute()
            )
            for f in response.get("files", []):
                files.append(
                    DriveFileInfo(
                        id=f["id"],
                        name=f.get("name", ""),
                        mime_type=f.get("mimeType", ""),
                        md5_checksum=f.get("md5Checksum"),
                        size=int(f.get("size") or 0),
                        folder_id=folder_id,
                    )
                )
            token = response.get("nextPageToken")
            if not token:
                break
        return files

    def download(self, file_id: str) -> bytes:
        service = self._build()
        from googleapiclient.http import MediaIoBaseDownload

        request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return buffer.getvalue()


# ---------------------------------------------------------------------------
# In-memory client for mock mode and tests
# ---------------------------------------------------------------------------


class InMemoryDriveClient:
    """A Drive stand-in backed by a dict of folders -> (filename, bytes)."""

    def __init__(self, folders: dict[str, dict[str, Any]] | None = None, *, email: str = "mock-sa@example.iam.gserviceaccount.com"):
        self._folders: dict[str, dict[str, Any]] = folders or {}
        self._email = email
        self.download_calls: list[str] = []

    # -- population helpers -------------------------------------------------
    def add_folder(self, folder_id: str, name: str) -> None:
        self._folders.setdefault(folder_id, {"name": name, "files": {}})

    def add_file(
        self,
        folder_id: str,
        file_id: str,
        filename: str,
        data: bytes,
        *,
        mime_type: str | None = None,
    ) -> DriveFileInfo:
        if folder_id not in self._folders:
            self.add_folder(folder_id, folder_id)
        mime = mime_type or (PDF_MIME if filename.lower().endswith(".pdf") else DOCX_MIME)
        info = DriveFileInfo(
            id=file_id,
            name=filename,
            mime_type=mime,
            md5_checksum=hashlib.md5(data).hexdigest(),
            size=len(data),
            folder_id=folder_id,
        )
        self._folders[folder_id]["files"][file_id] = (info, data)
        return info

    def load_directory(self, folder_id: str, name: str, directory: Path) -> None:
        """Register every supported file in a local directory as Drive content."""
        self.add_folder(folder_id, name)
        for path in sorted(Path(directory).iterdir()):
            if path.suffix.lower() in {".pdf", ".docx"}:
                self.add_file(folder_id, f"{folder_id}:{path.name}", path.name, path.read_bytes())

    # -- DriveClient protocol -----------------------------------------------
    def status(self) -> dict[str, Any]:
        return {
            "connected": True,
            "service_account_email": self._email,
            "error": None,
            "mode": "mock",
        }

    def list_folders(self) -> list[DriveFolderInfo]:
        return [DriveFolderInfo(fid, meta["name"]) for fid, meta in self._folders.items()]

    def list_files(self, folder_id: str) -> list[DriveFileInfo]:
        meta = self._folders.get(folder_id)
        if meta is None:
            return []
        return [info for info, _ in meta["files"].values()]

    def download(self, file_id: str) -> bytes:
        self.download_calls.append(file_id)
        for meta in self._folders.values():
            if file_id in meta["files"]:
                return meta["files"][file_id][1]
        raise FileNotFoundError(f"no such Drive file: {file_id}")


def compute_md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def iter_supported(files: Iterable[DriveFileInfo]) -> list[DriveFileInfo]:
    return [f for f in files if f.mime_type in SUPPORTED_MIMES or f.name.lower().endswith((".pdf", ".docx"))]
