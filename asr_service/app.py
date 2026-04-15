"""
Dedicated ASR worker microservice.

Each service instance owns exactly one language model. The main API proxies
verbal-flashcard transcription requests to the worker that matches the user's
learned language.
"""

import io
import os
import tempfile
import threading
import time
from datetime import datetime

from flask import Flask, Response, jsonify, request

import psutil

try:
    from huggingface_hub import scan_cache_dir
except ImportError:
    scan_cache_dir = None


ASR_LANGUAGE_CODE = os.environ.get("ASR_LANGUAGE_CODE", "da").casefold()
ASR_MODEL_NAME = os.environ.get(
    "ASR_MODEL_NAME",
    "nvidia/parakeet-rnnt-110m-da-dk",
)
ASR_WORKER_NAME = os.environ.get(
    "ASR_WORKER_NAME",
    f"asr-{ASR_LANGUAGE_CODE}",
)

_stats_lock = threading.Lock()
_request_stats = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "last_request_at": None,
}
_last_request_metrics = None
_model_stats = {
    "configured_model_name": ASR_MODEL_NAME,
    "load_started_at": None,
    "load_finished_at": None,
    "load_duration_ms": None,
    "memory_before_bytes": None,
    "memory_after_bytes": None,
    "memory_delta_bytes": None,
    "cache_size_bytes": None,
}


def _get_process_memory_stats():
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        "rss_bytes": memory_info.rss,
        "rss_mb": round(memory_info.rss / 1024 / 1024, 1),
        "memory_percent": round(process.memory_percent(), 1),
    }


def _get_hf_cached_model_size_bytes(model_name):
    if scan_cache_dir is None or not model_name:
        return None

    try:
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id == model_name:
                return repo.size_on_disk
    except Exception as exc:
        print(f"Could not inspect Hugging Face cache for {model_name}: {exc}")

    return None


def _bytes_to_mb(value):
    if value is None:
        return None
    return round(value / 1024 / 1024, 1)


try:
    import nemo.collections.asr as nemo_asr
    from pydub import AudioSegment

    ASR_AVAILABLE = True
    _memory_before = _get_process_memory_stats()
    _model_stats["load_started_at"] = datetime.now().isoformat()
    _load_started_at = time.perf_counter()
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=ASR_MODEL_NAME)
    _load_finished_at = time.perf_counter()
    _memory_after = _get_process_memory_stats()

    _model_stats["load_finished_at"] = datetime.now().isoformat()
    _model_stats["load_duration_ms"] = round(
        (_load_finished_at - _load_started_at) * 1000, 1
    )
    _model_stats["memory_before_bytes"] = _memory_before["rss_bytes"]
    _model_stats["memory_after_bytes"] = _memory_after["rss_bytes"]
    if _memory_before["rss_bytes"] is not None and _memory_after["rss_bytes"] is not None:
        _model_stats["memory_delta_bytes"] = (
            _memory_after["rss_bytes"] - _memory_before["rss_bytes"]
        )
    _model_stats["cache_size_bytes"] = _get_hf_cached_model_size_bytes(ASR_MODEL_NAME)
    print(
        f"Loaded ASR worker {ASR_WORKER_NAME} for {ASR_LANGUAGE_CODE} "
        f"with model {ASR_MODEL_NAME}"
    )
except ImportError as exc:
    ASR_AVAILABLE = False
    asr_model = None
    print(f"ASR worker dependencies unavailable: {exc}")
except Exception as exc:
    ASR_AVAILABLE = False
    asr_model = None
    print(f"Failed to load ASR worker model {ASR_MODEL_NAME}: {exc}")


def _build_stats():
    process_memory = _get_process_memory_stats()
    return {
        "worker_name": ASR_WORKER_NAME,
        "worker_language": ASR_LANGUAGE_CODE,
        "configured_model_name": _model_stats.get("configured_model_name"),
        "asr_available": ASR_AVAILABLE,
        "model_loaded": asr_model is not None,
        "load_started_at": _model_stats.get("load_started_at"),
        "load_finished_at": _model_stats.get("load_finished_at"),
        "load_duration_ms": _model_stats.get("load_duration_ms"),
        "process_memory_rss_bytes": process_memory["rss_bytes"],
        "process_memory_rss_mb": process_memory["rss_mb"],
        "process_memory_percent": process_memory["memory_percent"],
        "memory_before_load_bytes": _model_stats.get("memory_before_bytes"),
        "memory_before_load_mb": _bytes_to_mb(_model_stats.get("memory_before_bytes")),
        "memory_after_load_bytes": _model_stats.get("memory_after_bytes"),
        "memory_after_load_mb": _bytes_to_mb(_model_stats.get("memory_after_bytes")),
        "memory_delta_bytes": _model_stats.get("memory_delta_bytes"),
        "memory_delta_mb": _bytes_to_mb(_model_stats.get("memory_delta_bytes")),
        "model_cache_size_bytes": _model_stats.get("cache_size_bytes"),
        "model_cache_size_mb": _bytes_to_mb(_model_stats.get("cache_size_bytes")),
        "request_counts": dict(_request_stats),
        "last_request_metrics": _last_request_metrics,
    }


def _finalize_request_metrics(metrics):
    global _last_request_metrics

    with _stats_lock:
        _request_stats["total_requests"] += 1
        _request_stats["last_request_at"] = metrics["request_started_at"]
        if metrics["status"] == "success":
            _request_stats["successful_requests"] += 1
        else:
            _request_stats["failed_requests"] += 1
        _last_request_metrics = metrics

    return metrics


def transcribe_audio_file(audio_storage, flashcard_id=None, requested_language_code=None):
    request_started_at = datetime.now().isoformat()
    started_at = time.perf_counter()
    process_memory_before = _get_process_memory_stats()
    audio_duration_ms = None
    audio_input_bytes = None
    wav_file_size_bytes = None
    transcription = None
    error_message = None
    temp_path = None

    if requested_language_code and requested_language_code.casefold() != ASR_LANGUAGE_CODE:
        raise ValueError(
            f"Worker {ASR_WORKER_NAME} handles '{ASR_LANGUAGE_CODE}', "
            f"not '{requested_language_code}'"
        )

    try:
        audio_bytes = audio_storage.read()
        audio_input_bytes = len(audio_bytes)

        if not ASR_AVAILABLE or asr_model is None:
            raise RuntimeError("ASR model is not available in this worker")

        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio_duration_ms = len(audio)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            audio.export(temp_path, format="wav")

        wav_file_size_bytes = os.path.getsize(temp_path)
        transcript = asr_model.transcribe([temp_path])

        if isinstance(transcript, tuple) and len(transcript) == 2:
            transcript = transcript[0]

        first = transcript[0]
        if hasattr(first, "text"):
            transcription = first.text
        elif isinstance(first, str):
            transcription = first
        elif isinstance(first, list) and first:
            nested = first[0]
            if hasattr(nested, "text"):
                transcription = nested.text
            elif isinstance(nested, str):
                transcription = nested

        if transcription is None:
            raise TypeError(
                f"Unexpected transcription output: {type(transcript)} / {type(first)}"
            )
    except Exception as exc:
        error_message = str(exc)
        raise
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

        process_memory_after = _get_process_memory_stats()
        process_memory_delta_bytes = None
        if (
            process_memory_before["rss_bytes"] is not None
            and process_memory_after["rss_bytes"] is not None
        ):
            process_memory_delta_bytes = (
                process_memory_after["rss_bytes"] - process_memory_before["rss_bytes"]
            )

        _finalize_request_metrics(
            {
                "request_started_at": request_started_at,
                "request_duration_ms": round((time.perf_counter() - started_at) * 1000, 1),
                "status": "error" if error_message else "success",
                "worker_name": ASR_WORKER_NAME,
                "worker_language": ASR_LANGUAGE_CODE,
                "configured_model_name": ASR_MODEL_NAME,
                "flashcard_id": flashcard_id,
                "audio_input_bytes": audio_input_bytes,
                "audio_duration_ms": audio_duration_ms,
                "wav_file_size_bytes": wav_file_size_bytes,
                "transcription_chars": len(transcription or ""),
                "error_message": error_message,
                "process_memory_before_bytes": process_memory_before["rss_bytes"],
                "process_memory_before_mb": process_memory_before["rss_mb"],
                "process_memory_after_bytes": process_memory_after["rss_bytes"],
                "process_memory_after_mb": process_memory_after["rss_mb"],
                "process_memory_delta_bytes": process_memory_delta_bytes,
                "process_memory_delta_mb": _bytes_to_mb(process_memory_delta_bytes),
            }
        )

    return transcription, _last_request_metrics


app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    stats = _build_stats()
    return jsonify(
        {
            "status": "ok" if stats["asr_available"] and stats["model_loaded"] else "degraded",
            "worker_name": stats["worker_name"],
            "worker_language": stats["worker_language"],
            "model_loaded": stats["model_loaded"],
            "process_memory_rss_mb": stats["process_memory_rss_mb"],
        }
    )


@app.route("/stats", methods=["GET"])
def stats():
    return jsonify(_build_stats())


@app.route("/metrics", methods=["GET"])
def metrics():
    stats = _build_stats()
    request_counts = stats.get("request_counts") or {}
    last_request = stats.get("last_request_metrics") or {}

    lines = [
        "# HELP asr_worker_available Whether ASR libraries are available",
        "# TYPE asr_worker_available gauge",
        f'asr_worker_available{{worker="{ASR_WORKER_NAME}",language="{ASR_LANGUAGE_CODE}"}} {1 if stats["asr_available"] else 0}',
        "",
        "# HELP asr_worker_model_loaded Whether the worker model is currently loaded",
        "# TYPE asr_worker_model_loaded gauge",
        f'asr_worker_model_loaded{{worker="{ASR_WORKER_NAME}",language="{ASR_LANGUAGE_CODE}"}} {1 if stats["model_loaded"] else 0}',
        "",
        "# HELP asr_worker_process_memory_bytes Current worker process RSS memory",
        "# TYPE asr_worker_process_memory_bytes gauge",
        f'asr_worker_process_memory_bytes{{worker="{ASR_WORKER_NAME}",language="{ASR_LANGUAGE_CODE}"}} {stats["process_memory_rss_bytes"] or 0}',
        "",
        "# HELP asr_worker_model_cache_size_bytes On-disk Hugging Face cache size for the loaded model",
        "# TYPE asr_worker_model_cache_size_bytes gauge",
        f'asr_worker_model_cache_size_bytes{{worker="{ASR_WORKER_NAME}",language="{ASR_LANGUAGE_CODE}"}} {stats["model_cache_size_bytes"] or 0}',
        "",
        "# HELP asr_worker_load_duration_ms Time spent loading the worker model at startup",
        "# TYPE asr_worker_load_duration_ms gauge",
        f'asr_worker_load_duration_ms{{worker="{ASR_WORKER_NAME}",language="{ASR_LANGUAGE_CODE}"}} {stats["load_duration_ms"] or 0}',
        "",
        "# HELP asr_worker_load_memory_delta_bytes Change in process RSS during worker model load",
        "# TYPE asr_worker_load_memory_delta_bytes gauge",
        f'asr_worker_load_memory_delta_bytes{{worker="{ASR_WORKER_NAME}",language="{ASR_LANGUAGE_CODE}"}} {stats["memory_delta_bytes"] or 0}',
        "",
        "# HELP asr_worker_requests_total Total worker transcription requests",
        "# TYPE asr_worker_requests_total counter",
        f'asr_worker_requests_total{{worker="{ASR_WORKER_NAME}",language="{ASR_LANGUAGE_CODE}"}} {request_counts.get("total_requests", 0)}',
        "",
        "# HELP asr_worker_requests_failed_total Failed worker transcription requests",
        "# TYPE asr_worker_requests_failed_total counter",
        f'asr_worker_requests_failed_total{{worker="{ASR_WORKER_NAME}",language="{ASR_LANGUAGE_CODE}"}} {request_counts.get("failed_requests", 0)}',
        "",
        "# HELP asr_worker_last_request_duration_ms Duration of the most recent worker transcription request",
        "# TYPE asr_worker_last_request_duration_ms gauge",
        f'asr_worker_last_request_duration_ms{{worker="{ASR_WORKER_NAME}",language="{ASR_LANGUAGE_CODE}"}} {last_request.get("request_duration_ms") or 0}',
        "",
        "# HELP asr_worker_last_request_memory_delta_bytes Memory delta for the most recent worker transcription request",
        "# TYPE asr_worker_last_request_memory_delta_bytes gauge",
        f'asr_worker_last_request_memory_delta_bytes{{worker="{ASR_WORKER_NAME}",language="{ASR_LANGUAGE_CODE}"}} {last_request.get("process_memory_delta_bytes") or 0}',
        "",
        "# HELP asr_worker_last_request_audio_input_bytes Uploaded audio size for the most recent worker transcription request",
        "# TYPE asr_worker_last_request_audio_input_bytes gauge",
        f'asr_worker_last_request_audio_input_bytes{{worker="{ASR_WORKER_NAME}",language="{ASR_LANGUAGE_CODE}"}} {last_request.get("audio_input_bytes") or 0}',
    ]

    return Response("\n".join(lines) + "\n", mimetype="text/plain")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["file"]
    if audio_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    flashcard_id = request.form.get("flashcard_id")
    requested_language_code = request.form.get("language_code")

    try:
        transcription, request_metrics = transcribe_audio_file(
            audio_file,
            flashcard_id=flashcard_id,
            requested_language_code=requested_language_code,
        )
        return jsonify(
            {
                "success": True,
                "transcription": transcription,
                "request_metrics": request_metrics,
            }
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("ASR_SERVICE_PORT", "5002")))
