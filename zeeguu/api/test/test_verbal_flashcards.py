from fixtures import (
    logged_in_client as client,
    add_one_bookmark,
    add_context_types,
    add_source_types,
)


def test_verbal_flashcards_only_returns_level_3_plus_words(client):
    add_context_types()
    add_source_types()

    bookmark_id = add_one_bookmark(client)

    from zeeguu.core.model.bookmark import Bookmark
    from zeeguu.core.model.db import db

    bookmark = Bookmark.find(bookmark_id)
    bookmark.user_word.level = 2
    db.session.commit()

    flashcards = client.get("/verbal_flashcards")
    assert flashcards["total"] == 0

    bookmark.user_word.level = 3
    db.session.commit()

    flashcards = client.get("/verbal_flashcards")

    assert flashcards["total"] == 1
    assert len(flashcards["flashcards"]) == 1
    assert flashcards["flashcards"][0]["bookmark_id"] == bookmark_id
    assert flashcards["flashcards"][0]["prompt"] == bookmark.user_word.meaning.origin.content
    assert flashcards["flashcards"][0]["answer"] == bookmark.user_word.meaning.translation.content
    assert flashcards["flashcards"][0]["expectedText"] == bookmark.user_word.meaning.translation.content


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
            "user_answer": bookmark.user_word.meaning.translation.content,
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
