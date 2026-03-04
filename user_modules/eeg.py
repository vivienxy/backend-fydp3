from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

DEFAULT_CHANNEL_MAPPING = {
    "CH1": "Cz",
    "CH2": "Pz",
    "CH3": "T7",
    "CH4": "T8",
    "CH5": "P7",
    "CH6": "P8",
    "CH7": "O1",
    "CH8": "O2",
}

WIN_N250 = (0.2, 0.3)
WIN_P300 = (0.25, 0.35)
CH_WINDOWS = {
    "Cz": WIN_P300,
    "Pz": WIN_P300,
    "P7": WIN_N250,
    "P8": WIN_N250,
    "O1": WIN_N250,
    "O2": WIN_N250,
}

_EVENT_LOCK = threading.Lock()
_LAST_EVENT_TS: float | None = None


@dataclass
class EEGStreamContext:
    inlet: Any
    channel_names: list[str]
    sfreq: float
    buffer_seconds: float = 8.0
    pull_timeout: float = 0.05
    _buffer: deque[tuple[float, np.ndarray]] = field(default_factory=deque)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _running: bool = field(default=True)
    _thread: threading.Thread | None = None

    def __post_init__(self) -> None:
        self._thread = threading.Thread(target=self._pull_loop, daemon=True)
        self._thread.start()

    def _pull_loop(self) -> None:
        while self._running:
            sample, ts = self.inlet.pull_sample(timeout=self.pull_timeout)
            if ts is None or sample is None:
                continue
            arr = np.asarray(sample, dtype=float)
            with self._lock:
                self._buffer.append((float(ts), arr))
                cutoff = float(ts) - self.buffer_seconds
                while self._buffer and self._buffer[0][0] < cutoff:
                    self._buffer.popleft()

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            if not self._buffer:
                return np.empty((0, len(self.channel_names))), np.empty((0,))
            ts = np.asarray([row[0] for row in self._buffer], dtype=float)
            data = np.asarray([row[1] for row in self._buffer], dtype=float)
        return data, ts



def connect_eeg() -> EEGStreamContext:
    from pylsl import StreamInlet, resolve_byprop

    stream_type = os.getenv("EEG_LSL_STREAM_TYPE", "EEG")
    stream_name = os.getenv("EEG_LSL_STREAM_NAME")
    timeout = float(os.getenv("EEG_LSL_RESOLVE_TIMEOUT", "5.0"))

    streams = []
    if stream_name:
        streams = resolve_byprop("name", stream_name, timeout=timeout)
    if not streams:
        streams = resolve_byprop("type", stream_type, timeout=timeout)
    if not streams:
        raise RuntimeError("No EEG LSL stream found")

    inlet = StreamInlet(streams[0], max_buflen=60, max_chunklen=32)
    info = inlet.info()

    ch_count = int(info.channel_count())
    sfreq = float(info.nominal_srate()) or float(os.getenv("EEG_DEFAULT_SFREQ", "250"))

    channel_names: list[str] = []
    ch = info.desc().child("channels").child("channel")
    while ch.name() == "channel":
        label = ch.child_value("label") or f"CH{len(channel_names) + 1}"
        channel_names.append(DEFAULT_CHANNEL_MAPPING.get(label.upper(), label))
        ch = ch.next_sibling()

    if not channel_names:
        channel_names = [DEFAULT_CHANNEL_MAPPING.get(f"CH{i+1}", f"CH{i+1}") for i in range(ch_count)]

    context = EEGStreamContext(
        inlet=inlet,
        channel_names=channel_names,
        sfreq=sfreq,
        buffer_seconds=float(os.getenv("EEG_BUFFER_SECONDS", "8.0")),
    )

    warmup = float(os.getenv("EEG_CONNECT_WARMUP_SECONDS", "0.4"))
    if warmup > 0:
        time.sleep(warmup)

    return context


def event_filter(event_lsl_timestamp: float) -> bool:
    global _LAST_EVENT_TS

    if not np.isfinite(event_lsl_timestamp):
        return False

    min_interval = float(os.getenv("EEG_EVENT_MIN_INTERVAL_SEC", "0.15"))
    with _EVENT_LOCK:
        if _LAST_EVENT_TS is not None and event_lsl_timestamp <= _LAST_EVENT_TS:
            return False
        if _LAST_EVENT_TS is not None and (event_lsl_timestamp - _LAST_EVENT_TS) < min_interval:
            return False
        _LAST_EVENT_TS = event_lsl_timestamp
    return True


def create_epoch(stream: EEGStreamContext, event_lsl_timestamp: float) -> dict[str, Any]:
    tmin = float(os.getenv("EEG_EPOCH_TMIN", "-1.5"))
    tmax = float(os.getenv("EEG_EPOCH_TMAX", "1.5"))

    data, ts = stream.snapshot()
    if ts.size == 0:
        raise RuntimeError("EEG buffer empty; no samples available for epoching")

    start = event_lsl_timestamp + tmin
    end = event_lsl_timestamp + tmax
    mask = (ts >= start) & (ts <= end)
    if not np.any(mask):
        raise RuntimeError("No EEG samples in requested epoch window")

    ep_ts = ts[mask]
    ep_data = data[mask]

    expected_samples = int(round((tmax - tmin) * stream.sfreq)) + 1
    if ep_data.shape[0] < max(8, int(0.7 * expected_samples)):
        raise RuntimeError("Insufficient EEG samples in epoch window")

    return {
        "data": ep_data.T,
        "timestamps": ep_ts,
        "channel_names": stream.channel_names,
        "sfreq": stream.sfreq,
        "event_lsl_timestamp": event_lsl_timestamp,
        "tmin": tmin,
        "tmax": tmax,
    }


def _bandpass_and_notch(data: np.ndarray, sfreq: float) -> np.ndarray:
    from scipy import signal

    nyq = sfreq * 0.5
    b, a = signal.butter(4, [1.0 / nyq, 40.0 / nyq], btype="bandpass")
    y = signal.filtfilt(b, a, data, axis=1)

    for notch in (60.0, 120.0):
        if notch >= nyq:
            continue
        b_notch, a_notch = signal.iirnotch(w0=notch / nyq, Q=30)
        y = signal.filtfilt(b_notch, a_notch, y, axis=1)
    return y


def _extract_epoch_features(epoch_data: np.ndarray, ch_names: list[str], sfreq: float, tmin: float) -> dict[str, float]:
    from scipy.stats import kurtosis, skew

    stat_order = ["mean", "median", "max", "min", "ptp", "std", "skew", "auc", "kurtosis"]
    channel_index = {name.upper(): idx for idx, name in enumerate(ch_names)}

    selected: dict[str, tuple[float, float]] = {}
    for ch, win in CH_WINDOWS.items():
        if ch.upper() in channel_index:
            selected[ch] = win
    if not selected:
        raise RuntimeError("None of required feature channels were found in epoch data")

    time_axis = tmin + np.arange(epoch_data.shape[1], dtype=float) / sfreq

    row: dict[str, float] = {}
    for ch, (w0, w1) in selected.items():
        idx = channel_index[ch.upper()]
        mask = (time_axis >= float(w0)) & (time_axis <= float(w1))
        x = epoch_data[idx, mask] * 1e6
        if x.size == 0:
            vals = {s: float("nan") for s in stat_order}
        else:
            vals = {
                "mean": float(np.mean(x)),
                "median": float(np.median(x)),
                "max": float(np.max(x)),
                "min": float(np.min(x)),
                "ptp": float(np.ptp(x)),
                "std": float(np.std(x, ddof=0)),
                "skew": float(skew(x, bias=False)) if x.size > 2 else float("nan"),
                "auc": float(np.trapezoid(x, dx=1.0 / sfreq)),
                "kurtosis": float(kurtosis(x, fisher=False, bias=False)) if x.size > 3 else float("nan"),
            }
        for s in stat_order:
            row[f"{ch}_{s}"] = vals[s]

    n250_channels = ["P7", "P8", "O1", "O2"]
    p300_channels = ["Cz", "Pz"]
    for group_name, group_channels in (("N250avg", n250_channels), ("P300avg", p300_channels)):
        present = [ch for ch in group_channels if f"{ch}_mean" in row]
        if not present:
            continue
        for s in stat_order:
            vals = [row[f"{ch}_{s}"] for ch in present]
            row[f"{group_name}_{s}"] = float(np.nanmean(vals))

    preferred_ch_order = ["Cz", "Pz", "P7", "P8", "O1", "O2"]
    ordered_keys = []
    for ch in preferred_ch_order:
        ordered_keys.extend([f"{ch}_{s}" for s in stat_order])
    ordered_keys.extend([f"N250avg_{s}" for s in stat_order])
    ordered_keys.extend([f"P300avg_{s}" for s in stat_order])

    return {k: row.get(k, np.nan) for k in ordered_keys}


def eeg_processing(epoch: dict[str, Any]) -> np.ndarray:
    data = np.asarray(epoch["data"], dtype=float)
    sfreq = float(epoch["sfreq"])
    ch_names = list(epoch["channel_names"])
    tmin = float(epoch.get("tmin", -1.5))

    filt = _bandpass_and_notch(data, sfreq)
    reref = filt - np.mean(filt, axis=0, keepdims=True)

    baseline_start, baseline_end = -1.0, -0.7
    time_axis = tmin + np.arange(reref.shape[1], dtype=float) / sfreq
    base_mask = (time_axis >= baseline_start) & (time_axis <= baseline_end)
    if np.any(base_mask):
        reref = reref - np.mean(reref[:, base_mask], axis=1, keepdims=True)

    ptp_uv = np.ptp(reref * 1e6, axis=1)
    amp_thresh_uv = float(os.getenv("EEG_PTP_REJECT_THRESHOLD_UV", "200"))
    if np.any(ptp_uv > amp_thresh_uv):
        raise RuntimeError("Epoch rejected due to high peak-to-peak amplitude")

    features = _extract_epoch_features(reref, ch_names, sfreq, tmin)
    return np.asarray([features[k] for k in features.keys()], dtype=float)
