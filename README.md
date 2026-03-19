# ADAD Python Backend Server

This repository implements only the Python backend server block from the architecture diagram. It orchestrates EEG-event processing, live face recognition, cue decisioning, and local face/cue database storage for AR integration.

## What this backend does

- Connects to EEG LSL on startup with automatic retry.
- Accepts AR event timestamps via REST and runs EEG pipeline only when events arrive.
- Accepts a live ML2 Camera 2.1 video stream descriptor over WebSocket and runs face recognition continuously from that stream in a background worker.
- Combines EEG result + latest face identity to run cue preparation and emit real-time cue decisions over WebSocket.
- Supports face database uploads (image + metadata) and cue database uploads (JSON).
- Serves face/cue manifests for AR startup sync and provides file download endpoint for stored images.

## Project layout

- `app/main.py`
  - FastAPI app creation and startup lifecycle.
  - REST routes for events, database upload/download, health checks.
  - WebSocket endpoints:
    - `/ws/ar` for cue decision output and optional DB sync push on connect.
    - `/ws/video` for incoming live ML2 Camera 2.1 stream descriptors from AR.
- `app/config.py`
  - Environment-based application settings using Pydantic settings.
- `app/state.py`
  - Shared mutable state with async locks and active video-stream configuration.
  - In-memory face DB, cue DB, EEG stream handle, and latest recognized face ID.
- `app/eeg_pipeline.py`
  - EEG connection retry loop.
  - Event-triggered EEG orchestration:
    - `event_filter -> create_epoch -> eeg_processing -> ml_classifier`.
  - Handles create_epoch signature compatibility when user function accepts either `(event_ts)` or `(stream, event_ts)`.
- `app/face_pipeline.py`
  - Validates ML2 Camera 2.1 as the only supported source.
  - Pulls and processes live video frames from a direct stream URL using OpenCV.
  - Background face recognition loop using the provided DNN function.
- `app/cue_service.py`
  - Calls the provided cue preparation function and shapes final cue decision payload.
- `app/storage/models.py`
  - Pydantic message/data schemas for events, live stream descriptors, cue decisions, and manifests.
- `app/storage/db.py`
  - Local disk persistence for face and cue manifests.
  - File writing for face images and secure file-path resolution for downloads.
- `requirements.txt`
  - Python dependencies.

## Hook points to your existing implementations

This backend expects these existing functions and imports:

- `from user_modules.eeg import connect_eeg, event_filter, create_epoch, eeg_processing`
- `from user_modules.model import ml_classifier`
- `from user_modules.face import dnn_face_recognition`
- `from user_modules.cue import cue_preparation`

If those imports are missing, the backend still starts but raises `NotImplementedError` when those functions are called.

## Message contracts

### Incoming event JSON `POST /events`

```json
{
  "event_id": "uuid-or-string",
  "event_lsl_timestamp": 12345.6789,
  "optional_context": {}
}
```

### Incoming live video stream JSON over `WS /ws/video`

```json
{
  "source": "ml2_camera_2_1",
  "stream_url": "<live-stream-url>",
  "is_live": true
}
```

### Outgoing cue decision JSON over `WS /ws/ar`

```json
{
  "type": "cue_decision",
  "payload": {
    "event_id": "...",
    "event_lsl_timestamp": 12345.6789,
    "face_id": "john_doe",
    "is_unfamiliar": true,
    "send_cue": true,
    "cue": {},
    "server_time": "2026-01-01T12:00:00+00:00"
  }
}
```

### DB sync timing

- Preferred pull model: AR fetches current data at startup using:
  - `GET /db/face`
  - `GET /db/cue`
- Push model (also implemented): backend sends `{"type":"db_sync", ...}` immediately when AR connects to `WS /ws/ar`.

## API summary

- `GET /health`
- `POST /events`
- `WS /ws/ar`
- `WS /ws/video`
- `POST /db/face` as multipart fields:
  - `face_id` (text)
  - `metadata_json` (text JSON)
  - `image` (file)
- `POST /db/cue` as multipart fields:
  - `face_id` (text)
  - `cue_json` (text JSON)
- `GET /db/face`
- `GET /db/cue`
- `GET /db/file?path=faces/<file_name>.jpg`

## Environment variables

- `EEG_LSL_RETRY_SECONDS=3`
- `DATA_DIR=./data`
- `VIDEO_MODE=ws`
- `VIDEO_PULL_URL=`
- `MAX_FRAME_QUEUE=32`
- `LOG_LEVEL=INFO`
- `MAX_UPLOAD_BYTES=5000000`
- `PEOPLE_JSON_PATH=../WebServer/database/PeopleDatabase/people.json`
- `IMAGES_DIR=../WebServer/database/PeopleDatabase/images`
- `AUDITORY_CUE_DIR=../WebServer/database/PeopleDatabase/auditory cues`
- `HEADSHOTS_DIR=../WebServer/database/PeopleDatabase/headshots`

## Run instructions

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the backend:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

4. AR startup flow:
   - Connect to `WS /ws/ar` to receive `db_sync` and real-time cue decisions.
   - Fetch `GET /db/face` and `GET /db/cue` for latest manifests.
   - Send events to `POST /events`.
   - Send the ML2 Camera 2.1 live stream descriptor to `WS /ws/video`; backend continuously pulls live frames from `stream_url`.

## Notes on robustness

- Event requests return a structured error if EEG stream is not connected.
- Invalid/non-live stream descriptors are rejected.
- Database manifests are loaded from disk on startup; missing files initialize as empty databases.
- Face recognition failures do not crash server loops.
