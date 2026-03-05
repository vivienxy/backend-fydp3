import asyncio
import base64
import json
import logging
import struct
import time
from typing import Any

import cv2
import numpy as np

from app.state import AppState
from app.storage.models import VideoFrameMessage

logger = logging.getLogger(__name__)

try:
    from user_modules.face import dnn_face_recognition
except ImportError:
    # TO DO: implement import from user face module
    def dnn_face_recognition(frame: Any, face_db: dict[str, Any]) -> str | None:
        raise NotImplementedError("TO DO: implement dnn_face_recognition import from user modules")


async def enqueue_frame(state: AppState, timestamp: float, data_b64: str, encoding: str) -> None:
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


async def enqueue_frame_packet(state: AppState, packet: bytes) -> None:
    """
    Magic Leap Unity live-stream packet formats supported:
    1) Raw JPEG bytes -> server timestamp is used.
    2) Framed bytes: b"TS64" + <little-endian float64 timestamp> + <jpeg bytes>.
    3) UTF-8 JSON bytes matching VideoFrameMessage schema.
    """
    if not packet:
        raise ValueError("Empty frame packet")

    if packet.startswith(b"TS64"):
        if len(packet) < 12:
            raise ValueError("TS64 packet too short")
        timestamp = struct.unpack("<d", packet[4:12])[0]
        await enqueue_frame_bytes(state, timestamp, packet[12:], encoding="jpeg")
        return

    if packet[:1] == b"{" and packet[-1:] == b"}":
        payload = json.loads(packet.decode("utf-8"))
        frame = VideoFrameMessage.model_validate(payload)
        await enqueue_frame(state, frame.timestamp, frame.data_b64, frame.encoding)
        return

    await enqueue_frame_bytes(state, time.time(), packet, encoding="jpeg")


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
            logger.info("Processed frame", extra={"frame_ts": timestamp, "face_id": face_id})
        except Exception:
            logger.exception("Face recognition loop error")
        finally:
            state.frame_queue.task_done()
