import asyncio
from typing import Any

from app.config import Settings
from app.storage.db import LocalDB
from app.storage.models import CueRecord, FaceRecord


class AppState:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.db = LocalDB(
            data_dir=settings.data_dir,
            people_json_path=settings.people_json_path,
            images_dir=settings.images_dir,
            headshots_dir=settings.headshots_dir,
            auditory_cue_dir=settings.auditory_cue_dir,
        )
        self.db.ensure_dirs()

        self.face_db_lock = asyncio.Lock()
        self.cue_db_lock = asyncio.Lock()
        self.eeg_stream_lock = asyncio.Lock()

        self.face_db: dict[str, FaceRecord] = self.db.load_face_db()
        self.cue_db: dict[str, CueRecord] = self.db.load_cue_db()

        self.eeg_stream: Any | None = None
        self.current_face_id: str | None = None
        self.latest_eeg_result: dict[str, Any] = {}
        self.video_source_lock = asyncio.Lock()
        self.video_stream_url: str | None = None
        self.video_is_live: bool = False

    async def set_current_face(self, face_id: str | None) -> None:
        self.current_face_id = face_id

    async def set_eeg_stream(self, stream: Any) -> None:
        async with self.eeg_stream_lock:
            self.eeg_stream = stream

    async def get_eeg_stream(self) -> Any | None:
        async with self.eeg_stream_lock:
            return self.eeg_stream

    async def set_video_source(self, stream_url: str, is_live: bool) -> None:
        async with self.video_source_lock:
            self.video_stream_url = stream_url
            self.video_is_live = is_live

    async def get_video_source(self) -> tuple[str | None, bool]:
        async with self.video_source_lock:
            return self.video_stream_url, self.video_is_live
