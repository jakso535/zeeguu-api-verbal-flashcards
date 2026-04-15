import io
from werkzeug.datastructures import FileStorage

from fixtures import (
    logged_in_client as client,
    add_one_bookmark,
    add_context_types,
    add_source_types,
)


def _set_bookmark_level(bookmark_id, level):
    from zeeguu.core.model.bookmark import Bookmark
    from zeeguu.core.model.db import db

    bookmark = Bookmark.find(bookmark_id)
    bookmark.user_word.level = level
    db.session.commit()
    return bookmark


def _create_level_3_flashcard(client, word="hinter", translation="behind"):
    from zeeguu.core.model.context_identifier import ContextIdentifier
    from zeeguu.core.model.context_type import ContextType
    from zeeguu.core.model.bookmark import Bookmark
    from zeeguu.core.model.db import db
    from fixtures import create_and_get_article
    from zeeguu.core.test.mocking_the_web import URL_SPIEGEL_VENEZUELA

    article = create_and_get_article(client)
    context_i = ContextIdentifier(ContextType.ARTICLE_TITLE, None, article["id"])
    bookmark = client.post(
        "/contribute_translation/de/en",
        json={
            "word": word,
            "translation": translation,
            "context": f"stellt sich {word} Präsident",
            "url": URL_SPIEGEL_VENEZUELA,
            "source_id": article["source_id"],
            "context_identifier": context_i.as_dictionary(),
        },
    )
    bookmark_id = bookmark["bookmark_id"]
    bookmark_row = Bookmark.find(bookmark_id)
    bookmark_row.user_word.level = 3
    db.session.commit()
    return bookmark_row


def test_verbal_flashcards_only_returns_level_3_plus_words(client):
    add_context_types()
    add_source_types()

    bookmark_id = add_one_bookmark(client)
    bookmark = _set_bookmark_level(bookmark_id, 2)

    flashcards = client.get("/verbal_flashcards")
    assert flashcards["total"] == 0

    bookmark = _set_bookmark_level(bookmark_id, 3)
    expected_prompt = bookmark.user_word.meaning.translation.content
    expected_answer = bookmark.user_word.meaning.origin.content
    flashcards = client.get("/verbal_flashcards")

    assert flashcards["total"] == 1
    assert len(flashcards["flashcards"]) == 1
    assert flashcards["flashcards"][0]["bookmark_id"] == bookmark_id
    assert flashcards["flashcards"][0]["prompt"] == expected_prompt
    assert flashcards["flashcards"][0]["answer"] == expected_answer
    assert flashcards["flashcards"][0]["expectedText"] == expected_answer


def test_verbal_flashcards_deduplicate_same_origin_word(client):
    add_context_types()
    add_source_types()

    _create_level_3_flashcard(client, word="hinter", translation="behind")
    _create_level_3_flashcard(client, word="hinter", translation="at the back of")

    flashcards = client.get("/verbal_flashcards")

    assert flashcards["total"] == 1
    assert flashcards["flashcards"][0]["answer"] == "hinter"


def test_verbal_flashcards_paginates_results(client):
    add_context_types()
    add_source_types()

    _create_level_3_flashcard(client, word="hinter", translation="behind")
    _create_level_3_flashcard(client, word="gehen", translation="go")

    first_page = client.get("/verbal_flashcards?limit=1&offset=0")
    second_page = client.get("/verbal_flashcards?limit=1&offset=1")

    assert first_page["total"] == 2
    assert first_page["limit"] == 1
    assert first_page["offset"] == 0
    assert len(first_page["flashcards"]) == 1

    assert second_page["total"] == 2
    assert second_page["limit"] == 1
    assert second_page["offset"] == 1
    assert len(second_page["flashcards"]) == 1
    assert first_page["flashcards"][0]["id"] != second_page["flashcards"][0]["id"]


def test_sanitize_spoken_text_keeps_danish_letters_and_normalizes_spacing():
    from zeeguu.api.endpoints.verbal_flashcards import sanitize_spoken_text

    sanitized = sanitize_spoken_text("  MåDér!!!\n  er\t'FÅR'?  ")

    assert sanitized == "mådér er 'får'"


def test_canonical_danish_form_normalizes_to_stable_danish_spellings():
    from zeeguu.api.endpoints.verbal_flashcards import canonical_danish_form

    assert canonical_danish_form("Maade") == "måde"
    assert canonical_danish_form("OeL") == "øl"
    assert canonical_danish_form("hvad") == "hvad"


def test_asr_tolerant_danish_form_folds_danish_letters_for_transcript_matching():
    from zeeguu.api.endpoints.verbal_flashcards import asr_tolerant_danish_form

    assert asr_tolerant_danish_form("træ") == "tre"
    assert asr_tolerant_danish_form("måde") == "made"
    assert asr_tolerant_danish_form("øl") == "ol"
    assert asr_tolerant_danish_form("hvad") == "va"


def test_score_word_match_accepts_common_danish_asr_variants():
    from zeeguu.api.endpoints.verbal_flashcards import score_word_match

    aa_variant = score_word_match("maade", "måde")
    asr_variant = score_word_match("tre", "træ")

    assert aa_variant["isMatch"] is True
    assert aa_variant["matchType"] == "normalized_exact"
    assert asr_variant["isMatch"] is True
    assert asr_variant["matchType"] == "normalized_exact"


def test_calculate_accuracy_ignores_word_order_and_matches_fuzzily():
    from zeeguu.api.endpoints.verbal_flashcards import calculate_accuracy

    result = calculate_accuracy("hund stor", "stor hund")

    assert result["isAccepted"] is True
    assert result["acceptedWordCount"] == 2
    assert result["acceptedAccuracy"] == 100
    assert result["accuracy"] == 100


def test_calculate_accuracy_marks_close_but_incorrect_words():
    from zeeguu.api.endpoints.verbal_flashcards import calculate_accuracy

    result = calculate_accuracy("sok kat", "bog kat")

    assert result["isAccepted"] is False
    assert result["acceptedWordCount"] == 1
    assert result["wordMatches"][0]["word"] == "bog"
    assert result["wordMatches"][0]["isCorrect"] is False
    assert result["wordMatches"][0]["isClose"] is False


def test_check_pronunciation_requires_both_fields(client):
    add_context_types()
    add_source_types()

    response = client.client.post(
        client.append_session("/verbal_flashcards/check_pronunciation"),
        json={"user_speech": "hej"},
    )

    assert response.status_code == 400
    assert b"user_speech and expected_text are required" in response.data


def test_check_pronunciation_returns_accuracy_analysis(client):
    add_context_types()
    add_source_types()

    response = client.post(
        "/verbal_flashcards/check_pronunciation",
        json={"user_speech": "tre", "expected_text": "tr\u00e6"},
    )

    assert response["isAccepted"] is True
    assert response["acceptedWordCount"] == 1
    assert response["wordMatches"][0]["matchType"] == "normalized_exact"


def test_parse_asr_service_urls_supports_multiple_language_workers():
    from zeeguu.core.audio_lessons.asr_service_client import parse_asr_service_urls

    mapping = parse_asr_service_urls(
        "da=http://asr-da:5002, de=http://asr-de:5002;fr=http://asr-fr:5002"
    )

    assert mapping == {
        "da": "http://asr-da:5002",
        "de": "http://asr-de:5002",
        "fr": "http://asr-fr:5002",
    }


def test_asr_stats_reports_memory_and_cache_data(client, monkeypatch):
    add_context_types()
    add_source_types()

    from zeeguu.api.endpoints import verbal_flashcards

    monkeypatch.setattr(
        verbal_flashcards,
        "get_asr_service_url",
        lambda language_code: f"http://asr-{language_code}:5002",
    )
    monkeypatch.setattr(
        verbal_flashcards,
        "get_configured_asr_languages",
        lambda: ["da", "de"],
    )
    monkeypatch.setattr(
        verbal_flashcards,
        "fetch_asr_worker_stats",
        lambda language_code: {
            "worker_name": f"asr-{language_code}",
            "worker_language": language_code,
            "configured_model_name": "nvidia/parakeet-rnnt-110m-da-dk",
            "asr_available": True,
            "model_loaded": True,
            "load_started_at": "2026-04-15T12:00:00",
            "load_finished_at": "2026-04-15T12:00:03",
            "load_duration_ms": 3210.5,
            "process_memory_rss_bytes": 7777,
            "process_memory_rss_mb": 7.6,
            "process_memory_percent": 1.2,
            "memory_before_load_bytes": 1000,
            "memory_before_load_mb": 1.0,
            "memory_after_load_bytes": 5000,
            "memory_after_load_mb": 4.9,
            "memory_delta_bytes": 4000,
            "memory_delta_mb": 0.0,
            "model_cache_size_bytes": 123456,
            "model_cache_size_mb": 0.1,
            "request_counts": {
                "total_requests": 4,
                "successful_requests": 4,
                "failed_requests": 0,
                "mock_requests": 0,
                "last_request_at": "2026-04-15T12:05:00",
            },
            "last_request_metrics": None,
        },
    )

    response = client.get("/verbal_flashcards/asr_stats")

    assert response["asr_available"] is True
    assert response["model_loaded"] is True
    assert response["configured_model_name"] == "nvidia/parakeet-rnnt-110m-da-dk"
    assert response["configured_worker_url"] == "http://asr-de:5002"
    assert response["worker_name"] == "asr-de"
    assert response["worker_language"] == "de"
    assert response["worker_status"] == "ok"
    assert response["process_memory_rss_bytes"] == 7777
    assert response["process_memory_rss_mb"] == 7.6
    assert response["memory_delta_bytes"] == 4000
    assert response["model_cache_size_bytes"] == 123456
    assert response["learned_language"] == "de"
    assert response["model_matches_learned_language"] is True
    assert response["configured_languages"] == ["da", "de"]
    assert response["request_counts"]["total_requests"] == 4
    assert response["last_request_metrics"] is None


def test_asr_stats_reports_missing_worker_configuration(client, monkeypatch):
    add_context_types()
    add_source_types()

    from zeeguu.api.endpoints import verbal_flashcards

    monkeypatch.setattr(verbal_flashcards, "get_asr_service_url", lambda language_code: None)
    monkeypatch.setattr(verbal_flashcards, "get_configured_asr_languages", lambda: ["da"])

    response = client.get("/verbal_flashcards/asr_stats")

    assert response["asr_available"] is False
    assert response["model_loaded"] is False
    assert response["worker_status"] == "not_configured"
    assert response["configured_worker_url"] is None
    assert response["learned_language"] == "de"


def test_asr_metrics_returns_prometheus_style_metrics(client, monkeypatch):
    add_context_types()
    add_source_types()

    from zeeguu.api.endpoints import verbal_flashcards

    monkeypatch.setattr(
        verbal_flashcards,
        "get_asr_stats_for_user",
        lambda user: {
            "asr_available": True,
            "model_loaded": True,
            "process_memory_rss_bytes": 999,
            "model_cache_size_bytes": 888,
            "load_duration_ms": 777,
            "memory_delta_bytes": 666,
            "request_counts": {
                "total_requests": 3,
                "failed_requests": 1,
                "mock_requests": 2,
            },
            "last_request_metrics": {
                "request_duration_ms": 555,
                "process_memory_delta_bytes": 444,
                "audio_input_bytes": 333,
            },
        },
    )

    response = client.client.get(client.append_session("/verbal_flashcards/asr_metrics"))
    body = response.data.decode("utf-8")

    assert response.status_code == 200
    assert "verbal_flashcards_asr_available 1" in body
    assert "verbal_flashcards_asr_model_loaded 1" in body
    assert "verbal_flashcards_asr_process_memory_bytes 999" in body
    assert "verbal_flashcards_asr_model_cache_size_bytes 888" in body
    assert "verbal_flashcards_asr_load_duration_ms 777" in body
    assert "verbal_flashcards_asr_load_memory_delta_bytes 666" in body
    assert "verbal_flashcards_asr_requests_total 3" in body
    assert "verbal_flashcards_asr_requests_failed_total 1" in body
    assert "verbal_flashcards_asr_requests_mock_total 2" in body
    assert "verbal_flashcards_asr_last_request_duration_ms 555" in body
    assert "verbal_flashcards_asr_last_request_memory_delta_bytes 444" in body
    assert "verbal_flashcards_asr_last_request_audio_input_bytes 333" in body


def test_transcribe_endpoint_returns_request_metrics(client, monkeypatch):
    add_context_types()
    add_source_types()

    monkeypatch.setattr(
        "zeeguu.api.endpoints.verbal_flashcards.transcribe_audio",
        lambda audio_file, language_code=None, flashcard_id=None: {
            "transcription": "hej",
            "request_metrics": {
                "language_code": language_code,
                "flashcard_id": flashcard_id,
                "request_duration_ms": 123.4,
                "audio_input_bytes": 321,
                "process_memory_delta_bytes": 456,
            },
        },
    )

    response = client.client.post(
        client.append_session("/verbal_flashcards/transcribe"),
        data={"file": (io.BytesIO(b"fake audio"), "sample.wav"), "flashcard_id": "17"},
        content_type="multipart/form-data",
    )

    data = response.get_json()

    assert response.status_code == 200
    assert data["transcription"] == "hej"
    assert data["request_metrics"]["language_code"] == "de"
    assert data["request_metrics"]["flashcard_id"] == "17"
    assert data["request_metrics"]["request_duration_ms"] == 123.4


def test_transcribe_audio_routes_to_language_worker(monkeypatch):
    from zeeguu.api.endpoints import verbal_flashcards

    captured = {}

    def fake_transcribe_with_asr_worker(
        audio_bytes,
        filename,
        content_type,
        language_code,
        flashcard_id=None,
        service_url_map=None,
        timeout=None,
    ):
        captured["audio_bytes"] = audio_bytes
        captured["filename"] = filename
        captured["content_type"] = content_type
        captured["language_code"] = language_code
        captured["flashcard_id"] = flashcard_id
        return {
            "transcription": "hej",
            "request_metrics": {
                "request_duration_ms": 222.2,
            },
        }

    monkeypatch.setattr(
        verbal_flashcards,
        "transcribe_with_asr_worker",
        fake_transcribe_with_asr_worker,
    )

    audio_file = FileStorage(
        stream=io.BytesIO(b"audio-bytes"),
        filename="sample.webm",
        content_type="audio/webm",
    )

    result = verbal_flashcards.transcribe_audio(
        audio_file,
        language_code="da",
        flashcard_id="17",
    )

    assert result["transcription"] == "hej"
    assert result["request_metrics"]["language_code"] == "da"
    assert result["request_metrics"]["flashcard_id"] == "17"
    assert captured["audio_bytes"] == b"audio-bytes"
    assert captured["filename"] == "sample.webm"
    assert captured["content_type"] == "audio/webm"
    assert captured["language_code"] == "da"
    assert captured["flashcard_id"] == "17"


def test_transcribe_endpoint_returns_503_when_worker_is_not_configured(client, monkeypatch):
    add_context_types()
    add_source_types()

    from zeeguu.core.audio_lessons.asr_service_client import ASRServiceNotConfigured

    def raise_not_configured(audio_file, language_code=None, flashcard_id=None):
        raise ASRServiceNotConfigured("No ASR worker configured for language 'de'")

    monkeypatch.setattr(
        "zeeguu.api.endpoints.verbal_flashcards.transcribe_audio",
        raise_not_configured,
    )

    response = client.client.post(
        client.append_session("/verbal_flashcards/transcribe"),
        data={"file": (io.BytesIO(b"fake audio"), "sample.wav"), "flashcard_id": "17"},
        content_type="multipart/form-data",
    )

    assert response.status_code == 503
    assert b"No ASR worker configured for language 'de'" in response.data


def test_verbal_flashcards_submit_reports_exercise_outcome(client):
    add_context_types()
    add_source_types()

    bookmark_id = add_one_bookmark(client)

    from zeeguu.core.model.bookmark import Bookmark
    from zeeguu.core.model.db import db
    from zeeguu.core.model.exercise import Exercise

    bookmark = Bookmark.find(bookmark_id)
    bookmark.user_word.level = 3
    db.session.commit()

    response = client.post(
        "/verbal_flashcards/submit",
        json={
            "flashcard_id": str(bookmark_id),
            "user_answer": bookmark.user_word.meaning.origin.content,
            "is_correct": True,
            "answer_source": "speech",
            "response_time_ms": 1500,
        },
    )

    assert response["success"] is True
    assert response["flashcard_id"] == str(bookmark_id)
    assert response["is_correct"] is True
    assert response["exercise_outcome"] == "C"

    exercise = Exercise.query.order_by(Exercise.id.desc()).first()
    assert exercise.user_word_id == bookmark.user_word_id
    assert exercise.source.source == "Verbal Flashcards"
    assert exercise.outcome.outcome == "C"
    assert exercise.solving_speed == 1500


def test_submit_uses_fuzzy_acceptance_to_override_is_correct(client):
    add_context_types()
    add_source_types()

    bookmark_id = add_one_bookmark(client)
    _set_bookmark_level(bookmark_id, 3)

    response = client.post(
        "/verbal_flashcards/submit",
        json={
            "flashcard_id": str(bookmark_id),
            "user_answer": "hintar",
            "is_correct": False,
            "answer_source": "speech",
            "response_time_ms": "2000",
        },
    )

    assert response["success"] is True
    assert response["is_correct"] is True
    assert response["exercise_outcome"] == "C"
    assert response["accuracy_analysis"]["isAccepted"] is True
    assert response["accuracy_analysis"]["wordMatches"][0]["matchType"] in {"fuzzy", "normalized_exact"}
    assert response["flashcard_id"] == str(bookmark_id)


def test_submit_rejects_non_integer_session_id(client):
    add_context_types()
    add_source_types()

    bookmark_id = add_one_bookmark(client)
    _set_bookmark_level(bookmark_id, 3)

    response = client.client.post(
        client.append_session("/verbal_flashcards/submit"),
        json={
            "flashcard_id": str(bookmark_id),
            "user_answer": "hinter",
            "is_correct": True,
            "session_id": "abc",
        },
    )

    assert response.status_code == 400
    assert b"session_id must be an integer" in response.data


def test_submit_coerces_invalid_response_time_to_zero(client):
    add_context_types()
    add_source_types()

    bookmark_id = add_one_bookmark(client)

    from zeeguu.core.model.exercise import Exercise

    bookmark = _set_bookmark_level(bookmark_id, 3)

    response = client.post(
        "/verbal_flashcards/submit",
        json={
            "flashcard_id": str(bookmark_id),
            "user_answer": bookmark.user_word.meaning.origin.content,
            "is_correct": True,
            "response_time_ms": "not-a-number",
        },
    )

    exercise = Exercise.query.order_by(Exercise.id.desc()).first()

    assert response["success"] is True
    assert exercise.solving_speed == 0


def test_submit_returns_404_for_non_flashcard_word(client):
    add_context_types()
    add_source_types()

    bookmark_id = add_one_bookmark(client)
    _set_bookmark_level(bookmark_id, 1)

    response = client.client.post(
        client.append_session("/verbal_flashcards/submit"),
        json={
            "flashcard_id": str(bookmark_id),
            "user_answer": "hinter",
            "is_correct": True,
        },
    )

    assert response.status_code == 404
    assert b"Flashcard not found" in response.data


def test_reseed_flashcards_adds_level_3_danish_words_for_current_user(client):
    from zeeguu.core.model.user import User
    from zeeguu.core.model.user_word import UserWord
    from zeeguu.core.word_scheduling.basicSR.four_levels_per_word import FourLevelsPerWord

    response = client.post(
        "/verbal_flashcards/reseed",
        json={"count": 3},
    )

    assert response["success"] is True
    assert response["count"] == 3
    assert response["seeded_count"] == 3
    assert response["refreshed_count"] == 0
    assert response["total_selected"] == 3

    user = User.find(client.email)
    seeded_words = (
        UserWord.query.filter_by(user_id=user.id)
        .order_by(UserWord.id.asc())
        .all()
    )

    assert len(seeded_words) == 3
    assert all((user_word.level or 0) == 3 for user_word in seeded_words)
    assert all(user_word.fit_for_study is True for user_word in seeded_words)
    assert all(user_word.preferred_bookmark is not None for user_word in seeded_words)
    assert all(user_word.meaning.origin.language.code == "da" for user_word in seeded_words)
    assert all(user_word.meaning.translation.language.code == "en" for user_word in seeded_words)
    assert all(FourLevelsPerWord.find(user_word) is not None for user_word in seeded_words)


def test_seed_verbal_flashcards_refreshes_existing_words_when_bank_is_exhausted(client, monkeypatch):
    from zeeguu.core.exercises import verbal_flashcard_seeding
    from zeeguu.core.model.user import User
    from zeeguu.core.model.db import db

    monkeypatch.setattr(
        verbal_flashcard_seeding,
        "VERBAL_FLASHCARD_TEST_WORDS",
        [("hus", "house")],
    )

    user = User.find(client.email)

    first_result = verbal_flashcard_seeding.seed_verbal_flashcards_for_user(
        db.session, user, count=1
    )
    second_result = verbal_flashcard_seeding.seed_verbal_flashcards_for_user(
        db.session, user, count=1
    )

    assert first_result["seeded_count"] == 1
    assert first_result["refreshed_count"] == 0
    assert second_result["seeded_count"] == 0
    assert second_result["refreshed_count"] == 1
    assert second_result["refreshed_words"] == [{"origin": "hus", "translation": "house"}]
