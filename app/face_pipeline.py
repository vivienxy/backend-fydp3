import asyncio
import base64
import logging
import os
import tempfile
import time
from typing import Any

import cv2
import numpy as np

from app.state import AppState

logger = logging.getLogger(__name__)

try:
    from user_modules.face import dnn_face_recognition
except ImportError:
    # TO DO: implement import from user face module
    def dnn_face_recognition(frame: Any, face_db: dict[str, Any]) -> str | None:
        raise NotImplementedError("TO DO: implement dnn_face_recognition import from user modules")


async def enqueue_frame(state: AppState, timestamp: float, data_b64: str, encoding: str) -> None:
    """Legacy single-frame path; retained for compatibility."""
    try:
        raw = base64.b64decode(data_b64, validate=True)
    except Exception as exc:
        raise ValueError("Malformed base64 frame data") from exc
    await enqueue_frame_bytes(state, timestamp, raw, encoding)


async def enqueue_frame_bytes(state: AppState, timestamp: float, image_bytes: bytes, encoding: str = "jpeg") -> None:
    if encoding.lower() not in {"jpeg", "jpg"}:
        raise ValueError("Only jpeg encoding is supported")
    if len(image_bytes) > state.settings.max_video_frame_bytes:
        raise ValueError("Frame payload too large")

    if state.frame_queue.full():
        _ = state.frame_queue.get_nowait()
    await state.frame_queue.put((float(timestamp), image_bytes))


def _decode_video_chunk(video_bytes: bytes, container: str, target_fps: float) -> list[tuple[float, bytes]]:
    if len(video_bytes) == 0:
        return []

    suffix = f".{container.lower().strip('.') or 'mp4'}"
    frames: list[tuple[float, bytes]] = []

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ValueError("Unable to open incoming video chunk")

        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if not src_fps or src_fps <= 0:
            src_fps = target_fps if target_fps > 0 else 10.0

        stride = max(1, int(round(src_fps / max(target_fps, 0.1))))
        index = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % stride == 0:
                ok_jpg, enc = cv2.imencode('.jpg', frame)
                if ok_jpg:
                    ts = index / float(src_fps)
                    frames.append((ts, enc.tobytes()))
            index += 1

        cap.release()
        return frames
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


async def ingest_video_chunk(
    state: AppState,
    chunk_bytes: bytes,
    *,
    timestamp: float | None = None,
    container: str = "mp4",
) -> int:
    """
    Decode a video stream chunk (mp4/webm/etc) and enqueue sampled JPEG frames.
    This is the preferred path for ML2 live camera stream transport.
    """
    if len(chunk_bytes) > state.settings.max_video_chunk_bytes:
        raise ValueError("Video chunk payload too large")

    base_ts = float(timestamp) if timestamp is not None else time.time()
    target_fps = float(state.settings.video_sample_fps)

    decoded = await asyncio.to_thread(_decode_video_chunk, chunk_bytes, container, target_fps)
    for rel_ts, jpeg in decoded:
        await enqueue_frame_bytes(state, base_ts + rel_ts, jpeg, encoding="jpeg")
    return len(decoded)


def _decode_jpeg(image_bytes: bytes) -> np.ndarray:
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Invalid jpeg frame")
    return frame


async def face_recognition_loop(state: AppState) -> None:
    while True:
        timestamp, image_bytes = await state.frame_queue.get()
        try:
            frame = await asyncio.to_thread(_decode_jpeg, image_bytes)
            async with state.face_db_lock:
                face_db_view = {k: v.model_dump(mode="json") for k, v in state.face_db.items()}
            face_id = await asyncio.to_thread(dnn_face_recognition, frame, face_db_view)
            await state.set_current_face(face_id)
            logger.info("Processed stream frame", extra={"frame_ts": timestamp, "face_id": face_id})
        except Exception:
            logger.exception("Face recognition loop error")
        finally:
            state.frame_queue.task_done()
