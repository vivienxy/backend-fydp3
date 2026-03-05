import asyncio
import logging
from typing import Any

import cv2

from app.state import AppState

logger = logging.getLogger(__name__)
ML2_CAMERA_SOURCE = "ml2_camera_2_1"

try:
    from user_modules.face import dnn_face_recognition
except ImportError:
    # TO DO: implement import from user face module
    def dnn_face_recognition(frame: Any, face_db: dict[str, Any]) -> str | None:
        raise NotImplementedError("TO DO: implement dnn_face_recognition import from user modules")


async def update_video_stream(state: AppState, source: str, stream_url: str, is_live: bool) -> None:
    if source.lower() != ML2_CAMERA_SOURCE:
        raise ValueError(f"Only {ML2_CAMERA_SOURCE} is supported")
    if not is_live:
        raise ValueError("Only live video streams are supported")
    await state.set_video_source(stream_url, is_live)


def _run_dnn(frame: Any, face_db_view: dict[str, Any]) -> str | None:
    return dnn_face_recognition(frame, face_db_view)


async def face_recognition_loop(state: AppState) -> None:
    active_stream_url: str | None = None
    capture: cv2.VideoCapture | None = None

    while True:
        stream_url, is_live = await state.get_video_source()

        if not stream_url or not is_live:
            await asyncio.sleep(0.25)
            continue

        if stream_url != active_stream_url or capture is None or not capture.isOpened():
            if capture is not None:
                capture.release()
            capture = cv2.VideoCapture(stream_url)
            active_stream_url = stream_url
            if not capture.isOpened():
                logger.error("Unable to open live video stream", extra={"stream_url": stream_url})
                await asyncio.sleep(1)
                continue
            logger.info("Connected to ML2 Camera 2.1 live stream", extra={"stream_url": stream_url})

        ok, frame = await asyncio.to_thread(capture.read)
        if not ok or frame is None:
            logger.warning("Dropped frame from live stream; reconnecting", extra={"stream_url": stream_url})
            capture.release()
            capture = None
            await asyncio.sleep(0.2)
            continue

        try:
            async with state.face_db_lock:
                face_db_view = {k: v.model_dump(mode="json") for k, v in state.face_db.items()}
            face_id = await asyncio.to_thread(_run_dnn, frame, face_db_view)
            await state.set_current_face(face_id)
            logger.info("Processed live video frame", extra={"source": ML2_CAMERA_SOURCE, "face_id": face_id})
        except Exception:
            logger.exception("Face recognition loop error")

