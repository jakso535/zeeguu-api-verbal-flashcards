import traceback
import flask
import io
import os
import tempfile
import re
import unicodedata
from datetime import datetime
from flask import request
from sqlalchemy.orm import joinedload

from zeeguu.core.model.user import User
from zeeguu.core.model.user_word import UserWord
from zeeguu.core.model.meaning import Meaning
from zeeguu.core.model.phrase import Phrase
from zeeguu.core.model.bookmark import Bookmark
from zeeguu.core.model.exercise_outcome import ExerciseOutcome
from zeeguu.core.word_scheduling.basicSR.basicSR import BasicSRSchedule
from zeeguu.core.word_scheduling.basicSR.four_levels_per_word import FourLevelsPerWord
from zeeguu.api.utils.route_wrappers import cross_domain, requires_session
from zeeguu.api.utils.json_result import json_result
from . import api, db_session
from zeeguu.logging import log

# Try to import ASR libraries, but make it optional
try:
    import nemo.collections.asr as nemo_asr
    from pydub import AudioSegment
    ASR_AVAILABLE = True
    # Load the ASR model once at module load
    asr_model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/parakeet-rnnt-110m-da-dk"
    )
    log("ASR model loaded successfully")
except ImportError as e:
    ASR_AVAILABLE = False
    asr_model = None
    log(f"ASR libraries not available: {e}")
except Exception as e:
    ASR_AVAILABLE = False
    asr_model = None
    log(f"Failed to load ASR model: {e}")


VERBAL_FLASHCARD_EXERCISE_SOURCE = "Verbal Flashcards"


def _verbal_flashcard_from_user_word(user_word):
    bookmark = user_word.preferred_bookmark
    if not bookmark:
        return None

    prompt = user_word.meaning.translation.content
    answer = user_word.meaning.origin.content

    if not prompt or not answer:
        return None

    return {
        "id": str(bookmark.id),
        "bookmark_id": bookmark.id,
        "user_word_id": user_word.id,
        "level": user_word.level,
        "from": user_word.meaning.origin.content,
        "to": user_word.meaning.translation.content,
        "origin": user_word.meaning.origin.content,
        "translation": user_word.meaning.translation.content,
        "prompt": prompt,
        "answer": answer,
        "expectedText": answer,
    }


def get_flashcard_collection(user):
    """
    Return level-3+ Zeeguu study words as minimal verbal flashcards.
    """
    user_words = BasicSRSchedule.user_words_to_study(user)
    flashcards = []
    seen_words = set()

    for user_word in user_words:
        if (user_word.level or 0) < 3:
            continue

        word_text = user_word.meaning.origin.content.lower()
        if word_text in seen_words:
            continue

        try:
            card = _verbal_flashcard_from_user_word(user_word)
        except Exception as e:
            log(f"Skipping verbal flashcard for user_word {user_word.id}: {e}")
            continue

        if card:
            seen_words.add(word_text)
            flashcards.append(card)

    return flashcards


def _ensure_schedule_for_verbal_flashcard(user_word):
    """
    Verbal flashcards can target higher-level words that are not currently in the
    standard exercise pipeline. Create a schedule row without resetting the level
    so the word appears in /words after it is practiced.
    """
    schedule = FourLevelsPerWord.find(user_word)
    if schedule:
        return schedule

    schedule = FourLevelsPerWord(user_word=user_word)
    schedule.next_practice_time = datetime.now()
    schedule.consecutive_correct_answers = 0
    schedule.cooling_interval = 0
    db_session.add(schedule)
    db_session.commit()
    return schedule


# ====================================
# Helper Functions
# ====================================
FUZZY_ACCEPTANCE_BUFFER = 0.08



def canonical_danish_form(word):
    """
    Normalize a word into a canonical written Danish form.

    This keeps Danish letters intact and only collapses common alternate
    spellings into their standard written forms.
    """
    if not word:
        return ""

    word = unicodedata.normalize("NFC", str(word).casefold())

    spelling_variants = {
        'aa': 'å',
        'ae': 'æ',
        'oe': 'ø',
    }

    for pattern, replacement in spelling_variants.items():
        word = word.replace(pattern, replacement)

    return word


def asr_tolerant_danish_form(word):
    """
    Fold a word into a more ASR-tolerant comparison form.

    This starts from the canonical written form, then applies permissive
    simplifications that help match common ASR spellings to the expected word.
    """
    word = canonical_danish_form(word)

    if word.startswith('hv'):
        word = 'v' + word[2:]

    if word.endswith('d'):
        word = word[:-1]
    if word.endswith('g'):
        word = word[:-1]

    asr_variants = {
        'æ': 'e',
        'ø': 'o',
        'å': 'a',
    }

    for pattern, replacement in asr_variants.items():
        word = word.replace(pattern, replacement)

    return word


def sanitize_spoken_text(text):

    """Keep Danish characters while normalizing whitespace and punctuation."""
    text = text.lower().strip() if text else ""
    text = re.sub(r"[^\w\sæøåÆØÅ']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def damerau_levenshtein_distance(source, target):
    """Classic dynamic-programming Damerau-Levenshtein distance."""
    if source == target:
        return 0

    source_length = len(source)
    target_length = len(target)

    if source_length == 0:
        return target_length
    if target_length == 0:
        return source_length

    distance = {}
    for i in range(-1, source_length + 1):
        distance[(i, -1)] = i + 1
    for j in range(-1, target_length + 1):
        distance[(-1, j)] = j + 1

    for i in range(source_length):
        for j in range(target_length):
            substitution_cost = 0 if source[i] == target[j] else 1
            distance[(i, j)] = min(
                distance[(i - 1, j)] + 1,
                distance[(i, j - 1)] + 1,
                distance[(i - 1, j - 1)] + substitution_cost,
            )

            if i > 0 and j > 0 and source[i] == target[j - 1] and source[i - 1] == target[j]:
                distance[(i, j)] = min(
                    distance[(i, j)],
                    distance[(i - 2, j - 2)] + substitution_cost,
                )

    return distance[(source_length - 1, target_length - 1)]


def normalized_damerau_levenshtein_similarity(source, target):
    """Return a similarity score in the range [0, 1]."""
    if not source and not target:
        return 1.0
    if not source or not target:
        return 0.0

    max_length = max(len(source), len(target))
    distance = damerau_levenshtein_distance(source, target)
    return max(0.0, 1.0 - (distance / max_length))


def jaro_similarity(source, target):
    """Return the Jaro similarity in the range [0, 1]."""
    if source == target:
        return 1.0

    source_length = len(source)
    target_length = len(target)

    if source_length == 0 or target_length == 0:
        return 0.0

    match_distance = max(source_length, target_length) // 2 - 1
    source_matches = [False] * source_length
    target_matches = [False] * target_length
    matches = 0
    transpositions = 0

    for i in range(source_length):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, target_length)

        for j in range(start, end):
            if target_matches[j]:
                continue
            if source[i] != target[j]:
                continue

            source_matches[i] = True
            target_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    target_index = 0
    for i in range(source_length):
        if not source_matches[i]:
            continue

        while not target_matches[target_index]:
            target_index += 1

        if source[i] != target[target_index]:
            transpositions += 1

        target_index += 1

    return (
        (matches / source_length)
        + (matches / target_length)
        + ((matches - (transpositions / 2)) / matches)
    ) / 3


def jaro_winkler_similarity(source, target, prefix_weight=0.1):
    """Return the Jaro-Winkler similarity in the range [0, 1]."""
    similarity = jaro_similarity(source, target)
    common_prefix = 0

    for source_char, target_char in zip(source, target):
        if source_char != target_char or common_prefix == 4:
            break
        common_prefix += 1

    return similarity + (common_prefix * prefix_weight * (1 - similarity))


def boundary_aware_jaro_winkler_similarity(source, target):
    """
    Jaro-Winkler rewards shared prefixes. For ASR, also compare reversed strings
    so dropped initial sounds are not unfairly penalized.
    """
    if not source or not target:
        return 0.0

    forward_score = jaro_winkler_similarity(source, target)
    reversed_score = jaro_winkler_similarity(source[::-1], target[::-1])
    return max(forward_score, reversed_score)



def fuzzy_match_threshold(expected_word):
    """Length-aware thresholds tuned for short flashcard answers."""
    normalized_length = len(canonical_danish_form(expected_word))

    if normalized_length <= 2:
        return 1.0
    if normalized_length == 3:
        return 0.69
    if normalized_length == 4:
        return 0.76
    return 0.79



def score_word_match(user_word, expected_word):
    """Compare two words using exact, normalized, and fuzzy similarity signals."""
    user_word = user_word or ""
    expected_word = expected_word or ""

    normalized_user_word = canonical_danish_form(user_word)
    normalized_expected_word = canonical_danish_form(expected_word)
    asr_user_word = asr_tolerant_danish_form(user_word)
    asr_expected_word = asr_tolerant_danish_form(expected_word)

    if user_word == expected_word:
        return {
            "isMatch": True,
            "isExact": True,
            "matchType": "exact",
            "normalizedDamerauLevenshtein": 1.0,
            "jaroWinkler": 1.0,
            "combinedScore": 1.0,
            "matchThreshold": 1.0,
        }

    if (
        normalized_user_word == normalized_expected_word
        or asr_user_word == asr_expected_word
    ):
        return {
            "isMatch": True,
            "isExact": False,
            "matchType": "normalized_exact",
            "normalizedDamerauLevenshtein": 1.0,
            "jaroWinkler": 1.0,
            "combinedScore": 1.0,
            "matchThreshold": 1.0,
        }

    normalized_damerau_levenshtein = max(
        normalized_damerau_levenshtein_similarity(user_word, expected_word),
        normalized_damerau_levenshtein_similarity(normalized_user_word, normalized_expected_word),
        normalized_damerau_levenshtein_similarity(asr_user_word, asr_expected_word),
    )
    jaro_winkler = max(
        boundary_aware_jaro_winkler_similarity(user_word, expected_word),
        boundary_aware_jaro_winkler_similarity(normalized_user_word, normalized_expected_word),
        boundary_aware_jaro_winkler_similarity(asr_user_word, asr_expected_word),
    )

    combined_score = max(
        normalized_damerau_levenshtein,
        (normalized_damerau_levenshtein * 0.75) + (jaro_winkler * 0.25),
    )
    match_threshold = fuzzy_match_threshold(expected_word)

    return {
        "isMatch": combined_score >= match_threshold,
        "isExact": False,
        "matchType": "fuzzy" if combined_score >= match_threshold else "close",
        "normalizedDamerauLevenshtein": round(normalized_damerau_levenshtein, 3),
        "jaroWinkler": round(jaro_winkler, 3),
        "combinedScore": round(combined_score, 3),
        "matchThreshold": round(match_threshold, 3),
    }



def calculate_accuracy(user_speech, expected_text):
    """
    Calculate accuracy between user speech and expected text.
    Word order is intentionally ignored. Each expected word looks for the
    closest unmatched spoken word.
    """
    user_speech = sanitize_spoken_text(user_speech)
    expected_text = sanitize_spoken_text(expected_text)

    user_words = [w for w in user_speech.split() if len(w) > 0]
    expected_words = [w for w in expected_text.split() if len(w) > 0]

    word_matches = []
    accepted_words = 0
    matched_indices = set()
    word_score_total = 0.0

    for i, expected_word in enumerate(expected_words):
        best_candidate = None

        for j, user_word in enumerate(user_words):
            if j in matched_indices:
                continue

            scores = score_word_match(user_word, expected_word)
            candidate = {
                "userWord": user_word,
                "actualPosition": j,
                "scores": scores,
            }

            if best_candidate is None or scores["combinedScore"] > best_candidate["scores"]["combinedScore"]:
                best_candidate = candidate

        best_score = best_candidate["scores"] if best_candidate else None
        combined_score = best_score["combinedScore"] if best_score else 0.0
        is_match = bool(best_score and best_score["isMatch"])

        if is_match:
            matched_indices.add(best_candidate["actualPosition"])
            accepted_words += 1

        word_score_total += combined_score

        word_matches.append({
            "word": expected_word,
            "isCorrect": is_match,
            "userWord": best_candidate["userWord"] if best_candidate else None,
            "position": i,
            "suggestedWord": best_candidate["userWord"] if best_candidate else "?",
            "matchType": best_score["matchType"] if best_score else "missing",
            "normalizedDamerauLevenshtein": best_score["normalizedDamerauLevenshtein"] if best_score else 0.0,
            "jaroWinkler": best_score["jaroWinkler"] if best_score else 0.0,
            "combinedScore": round(combined_score, 3),
            "matchThreshold": best_score["matchThreshold"] if best_score else fuzzy_match_threshold(expected_word),
            "isClose": bool(best_score and combined_score >= (best_score["matchThreshold"] - FUZZY_ACCEPTANCE_BUFFER)),
        })

    word_accuracy = round((word_score_total / len(expected_words)) * 100) if expected_words else 0
    accepted_accuracy = round((accepted_words / len(expected_words)) * 100) if expected_words else 0
    is_accepted = bool(expected_words) and accepted_words == len(expected_words)

    feedback = get_feedback_message(word_accuracy, accepted_words, len(expected_words))
    detailed_analysis = generate_detailed_analysis(
        word_accuracy,
        accepted_words,
        len(expected_words),
        word_matches,
    )

    return {
        "accuracy": word_accuracy,
        "wordAccuracy": word_accuracy,
        "acceptedAccuracy": accepted_accuracy,
        "acceptedWordCount": accepted_words,
        "isAccepted": is_accepted,
        "feedback": feedback,
        "wordMatches": word_matches,
        "detailedAnalysis": detailed_analysis,
    }



def get_feedback_message(accuracy, accepted_words, total_words):
    """Generate feedback message without any word-order component."""
    if total_words and accepted_words == total_words:
        if accuracy >= 95:
            return "Excellent! Totally perfect! 🌟"
        if accuracy >= 85:
            return "Great! Almost perfect! ✨"
        return "Nice job! 👍"

    if accuracy >= 70:
        return "Close! Try once more 💪"
    if accuracy >= 50:
        return "Not bad! Keep practicing! 💪"
    if accuracy >= 30:
        return "Keep going! Try again 📚"
    if accuracy >= 10:
        return "Start slowly, say it clearly 🗣️"
    return "Try again, take your time 💪"


def generate_detailed_analysis(final_accuracy, correct_words, total_words, word_matches):
    """Generate detailed analysis without any word-order feedback."""
    if total_words == 0:
        return "No words to compare"

    incorrect_words = [w for w in word_matches if not w.get("isCorrect", False)]

    if len(incorrect_words) == 0:
        return f"Perfect! All {total_words} words were accepted. 🎉"

    if len(incorrect_words) == 1:
        close_word = incorrect_words[0]
        if close_word.get("isClose"):
            return (
                f"Very close. Expected '{close_word['word']}', "
                f"heard '{close_word.get('userWord') or '?'}'. Try once more."
            )
        return f"Keep practicing this word: '{close_word['word']}'"

    problem_words = ", ".join([f"'{w['word']}'" for w in incorrect_words[:3]])
    if len(incorrect_words) > 3:
        return (
            f"You got {correct_words} out of {total_words} words accepted. "
            f"Focus on: {problem_words} and {len(incorrect_words) - 3} more"
        )

    return f"You got {correct_words} out of {total_words} words accepted. Focus on: {problem_words}"


def transcribe_audio(audio_file):

    """
    Transcribe audio file using the ASR model.
    Returns the transcription text.
    """
    if not ASR_AVAILABLE or asr_model is None:
        # Mock transcription for testing
        log("ASR not available, returning mock transcription")
        return "Mock transcription: audio received"

    try:
        # Read and convert audio
        audio_data = audio_file.read()
        audio = AudioSegment.from_file(io.BytesIO(audio_data))
        audio = audio.set_channels(1).set_frame_rate(16000)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            audio.export(temp_path, format="wav")

        transcript = asr_model.transcribe([temp_path])

        # Clean up temp file
        os.unlink(temp_path)

        # TODO: research what type is expected from parakeet asr

        if isinstance(transcript, tuple) and len(transcript) == 2:
            transcript = transcript[0]

        first = transcript[0]

        if hasattr(first, "text"):
            return first.text
        if isinstance(first, str):
            return first
        if isinstance(first, list) and first:
            inner = first[0]
            if hasattr(inner, "text"):
                return inner.text
            if isinstance(inner, str):
                return inner

        raise TypeError(f"Unexpected transcription output: {type(transcript)} / {type(first)}")
    except Exception as e:
        log(f"Transcription error: {e}")
        raise


# ====================================
# API Endpoints
# ====================================

@api.route("/verbal_flashcards/transcribe", methods=["POST"])
@cross_domain
@requires_session
def transcribe_audio_endpoint():
    """
    Transcribe an audio recording for a verbal flashcard exercise.
    
    Expected form data:
    - file: audio file (required)
    - flashcard_id: optional ID of the current flashcard
    
    Returns:
    {
        "transcription": "transcribed text",
        "flashcard": {...}  # if flashcard_id provided
    }
    """
    try:
        # Get the uploaded file
        if 'file' not in request.files:
            return json_result({"error": "No audio file provided"}), 400

        audio_file = request.files['file']
        if audio_file.filename == '':
            return json_result({"error": "Empty filename"}), 400

        # Get optional flashcard_id
        flashcard_id = request.form.get('flashcard_id')

        # Transcribe the audio
        transcription = transcribe_audio(audio_file)

        # Get flashcard info if requested
        flashcard = None
        if flashcard_id:
            user = User.find_by_id(flask.g.user_id)
            flashcards = get_flashcard_collection(user)
            flashcard = next((f for f in flashcards if f['id'] == flashcard_id), None)

        # Log user activity
        user = User.find_by_id(flask.g.user_id)
        log(f"User {user.id} transcribed audio for flashcard {flashcard_id}")

        return json_result({
            "success": True,
            "transcription": transcription,
            "flashcard": flashcard
        })

    except Exception as e:
        log(f"Transcription endpoint error: {e}")
        traceback.print_exc()
        return json_result({"error": str(e)}), 500


@api.route("/verbal_flashcards", methods=["GET"])
@cross_domain
@requires_session
def get_flashcards():
    """
    Get flashcards.
    
    Query parameters:
    - limit: max number of cards to return (optional, default 50)
    - offset: pagination offset (optional, default 0)
    
    Returns list of flashcards.
    """
    try:
        # Get query parameters
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))

        # Get all flashcards
        user = User.find_by_id(flask.g.user_id)
        flashcards = get_flashcard_collection(user)

        # Apply pagination
        total = len(flashcards)
        paginated = flashcards[offset:offset + limit]

        # Log user activity
        log(f"User {user.id} requested flashcards")

        return json_result({
            "flashcards": paginated,
            "total": total,
            "limit": limit,
            "offset": offset
        })

    except Exception as e:
        log(f"Get flashcards error: {e}")
        traceback.print_exc()
        return json_result({"error": str(e)}), 500


@api.route("/verbal_flashcards/<flashcard_id>", methods=["GET"])
@cross_domain
@requires_session
def get_flashcard_by_id(flashcard_id):
    """
    Get a single flashcard by ID.
    
    Returns the flashcard object.
    """
    try:
        user = User.find_by_id(flask.g.user_id)
        flashcards = get_flashcard_collection(user)
        flashcard = next((f for f in flashcards if f['id'] == flashcard_id), None)

        if not flashcard:
            return json_result({"error": "Flashcard not found"}), 404

        # Log user activity
        log(f"User {user.id} requested flashcard {flashcard_id}")

        return json_result(flashcard)

    except Exception as e:
        log(f"Get flashcard error: {e}")
        traceback.print_exc()
        return json_result({"error": str(e)}), 500


@api.route("/verbal_flashcards/practice", methods=["GET"])
@cross_domain
@requires_session
def get_practice_set():
    """
    Get a set of flashcards for practice.
    Uses spaced repetition to select appropriate cards.
    
    Query parameters:
    - count: number of cards to return (optional, default 10)
    
    Returns list of flashcards for practice.
    """
    try:
        count = int(request.args.get('count', 10))
        user = User.find_by_id(flask.g.user_id)

        flashcards = get_flashcard_collection(user)
        practice_cards = flashcards[:count]

        log(f"User {user.id} requested practice set of size {count}")

        return json_result({
            "flashcards": practice_cards,
            "count": len(practice_cards)
        })

    except Exception as e:
        log(f"Get practice set error: {e}")
        traceback.print_exc()
        return json_result({"error": str(e)}), 500


@api.route("/verbal_flashcards/submit", methods=["POST"])
@cross_domain
@requires_session
def submit_answer():
    """
    Submit an answer for a flashcard and record performance.
    
    Expected JSON body:
    {
        "flashcard_id": "1",
        "user_answer": "transcribed text or typed answer",
        "is_correct": true/false,
        "answer_source": "speech|typing",
        "response_time_ms": 5000,
        "session_id": 123
    }
    
    Returns updated user progress and accuracy analysis.
    """
    try:
        data = request.get_json()
        if not data:
            return json_result({"error": "JSON body required"}), 400

        flashcard_id = str(data.get('flashcard_id')) if data.get('flashcard_id') is not None else None
        user_answer = data.get('user_answer', '')
        is_correct = data.get('is_correct')
        answer_source = data.get('answer_source', 'unknown')
        response_time = data.get('response_time_ms', 0)
        session_id = data.get('session_id')

        if not flashcard_id or is_correct is None:
            return json_result({"error": "flashcard_id and is_correct are required"}), 400

        user = User.find_by_id(flask.g.user_id)
        flashcards = get_flashcard_collection(user)
        flashcard = next((f for f in flashcards if f['id'] == flashcard_id), None)

        if not flashcard:
            return json_result({"error": "Flashcard not found"}), 404

        if session_id is not None:
            try:
                session_id = int(session_id)
            except (TypeError, ValueError):
                return json_result({"error": "session_id must be an integer"}), 400

        try:
            response_time = int(response_time)
        except (TypeError, ValueError):
            response_time = 0

        # Calculate accuracy analysis if user_answer is provided
        accuracy_analysis = None
        if user_answer:
            expected_text = flashcard["expectedText"]
            accuracy_analysis = calculate_accuracy(user_answer, expected_text)

            # Override is_correct only when the fuzzy matcher accepts the answer.
            if accuracy_analysis.get('isAccepted'):
                is_correct = True

        exercise_outcome = ExerciseOutcome.CORRECT if is_correct else ExerciseOutcome.WRONG
        other_feedback = f"answer_source={answer_source}"
        flashcard_user_word_id = flashcard["user_word_id"]

        from zeeguu.core.model.user_word import UserWord
        user_word = UserWord.query.get(flashcard_user_word_id)
        if not user_word or user_word.user_id != user.id:
            return json_result({"error": "Flashcard not found"}), 404

        _ensure_schedule_for_verbal_flashcard(user_word)

        user_word.report_exercise_outcome(
            db_session,
            VERBAL_FLASHCARD_EXERCISE_SOURCE,
            exercise_outcome,
            response_time,
            session_id,
            other_feedback,
        )

        log(f"User {user.id} answered flashcard {flashcard_id}: correct={is_correct}, source={answer_source}, time={response_time}ms, answer='{user_answer}'")

        response_data = {
            "success": True,
            "flashcard_id": flashcard_id,
            "is_correct": is_correct,
            "exercise_outcome": exercise_outcome,
            "message": "Answer recorded"
        }

        if accuracy_analysis:
            response_data["accuracy_analysis"] = accuracy_analysis

        return json_result(response_data)

    except Exception as e:
        log(f"Submit answer error: {e}")
        traceback.print_exc()
        return json_result({"error": str(e)}), 500


@api.route("/verbal_flashcards/check_pronunciation", methods=["POST"])
@cross_domain
@requires_session
def check_pronunciation():
    """
    Check pronunciation of user's speech against expected text.
    Returns accuracy analysis without storing progress.
    
    Expected JSON body:
    {
        "user_speech": "transcribed text",
        "expected_text": "expected phrase"
    }
    """
    try:
        data = request.get_json()
        if not data:
            return json_result({"error": "JSON body required"}), 400

        user_speech = data.get('user_speech', '')
        expected_text = data.get('expected_text', '')

        if not user_speech or not expected_text:
            return json_result({"error": "user_speech and expected_text are required"}), 400

        accuracy_analysis = calculate_accuracy(user_speech, expected_text)

        return json_result(accuracy_analysis)

    except Exception as e:
        log(f"Check pronunciation error: {e}")
        traceback.print_exc()
        return json_result({"error": str(e)}), 500
