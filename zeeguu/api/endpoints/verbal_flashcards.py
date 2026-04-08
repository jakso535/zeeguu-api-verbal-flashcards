import traceback
import flask
import io
import os
import tempfile
import random
import re
from flask import request

from zeeguu.core.model.user import User
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


# ====================================
# Mock Flashcard Data (replace with database later)
# ====================================
def get_flashcard_collection():
    """Return the flashcard collection - replace with database query later"""
    return [
        {
            "id": "1",
            "prompt": "Hello, how are you?",
            "hint": "Hej, hvordan har du det",
            "expectedText": "hej hvordan har du det",
            "example": "Person A: Hello, how are you? Person B: I'm fine, thanks!",
            "phoneticHint": "hej hvordan har du det"
        },
        {
            "id": "2",
            "prompt": "It is really nice weather today",
            "hint": "Det er virkelig godt vejr i dag",
            "expectedText": "det er virkelig godt vejr i dag",
            "example": "Let's go for a walk, it is really nice weather today!",
            "phoneticHint": "det er virkelig godt vejr i dag"
        },
        {
            "id": "3",
            "prompt": "I want to order a coffee",
            "hint": "Jeg vil gerne bestille en kaffe",
            "expectedText": "jeg vil gerne bestille en kaffe",
            "example": "Excuse me, I want to order a coffee please.",
            "phoneticHint": "jeg vil gerne bestille en kaffe"
        },
        {
            "id": "4",
            "prompt": "Can you help me?",
            "hint": "Kan du hjælpe mig?",
            "expectedText": "kan du hjælpe mig",
            "example": "I'm lost, can you help me find the station?",
            "phoneticHint": "kan du hjælpe mig"
        },
        {
            "id": "5",
            "prompt": "When is the meeting?",
            "hint": "Hvornår er mødet?",
            "expectedText": "hvornår er mødet",
            "example": "When is the meeting scheduled for today?",
            "phoneticHint": "hvornår er mødet"
        },
        {
            "id": "6",
            "prompt": "Coffee",
            "hint": "kaffe",
            "expectedText": "kaffe",
            "example": "I would like a coffee with milk.",
            "phoneticHint": "kaffe"
        },
        {
            "id": "7",
            "prompt": "Cake",
            "hint": "kage",
            "expectedText": "kage",
            "example": "This cake is delicious!",
            "phoneticHint": "kage"
        },
        {
            "id": "8",
            "prompt": "Sausage",
            "hint": "pølse",
            "expectedText": "pølse",
            "example": "Danish hot dogs with sausage are famous.",
            "phoneticHint": "pølse"
        }
    ]


# ====================================
# Helper Functions
# ====================================
def normalize_danish_word(word):
    """Normalize Danish words for better comparison"""
    if not word:
        return ""

    word = word.lower()

    # Handle common Danish spelling variations
    variations = {
        'aa': 'å',
        'ae': 'æ',
        'oe': 'ø',
        'hv': 'v',  # "hv" in "hvornår" often sounds like "v"
    }

    # Replace common variations
    for pattern, replacement in variations.items():
        word = word.replace(pattern, replacement)

    # Handle soft D at end of words (common in Danish)
    if word.endswith('d'):
        word = word[:-1]

    return word


def calculate_accuracy(user_speech, expected_text):
    """
    Calculate accuracy between user speech and expected text.
    Returns detailed accuracy metrics.
    """
    user_speech = user_speech.lower().strip() if user_speech else ""
    expected_text = expected_text.lower().strip() if expected_text else ""

    # Preserve Danish characters while removing punctuation
    user_speech = re.sub(r'[^\w\sæøåÆØÅ\']', ' ', user_speech)
    expected_text = re.sub(r'[^\w\sæøåÆØÅ\']', ' ', expected_text)

    # Normalize multiple spaces
    user_speech = re.sub(r'\s+', ' ', user_speech).strip()
    expected_text = re.sub(r'\s+', ' ', expected_text).strip()

    user_words = [w for w in user_speech.split() if len(w) > 0]
    expected_words = [w for w in expected_text.split() if len(w) > 0]

    word_matches = []
    correct_words = 0
    correct_positions = 0
    matched_indices = set()

    # First pass: find exact position matches
    for i in range(min(len(user_words), len(expected_words))):
        user_word = user_words[i]
        expected_word = expected_words[i]

        if (user_word == expected_word or
                normalize_danish_word(user_word) == normalize_danish_word(expected_word)):
            correct_words += 1
            correct_positions += 1
            matched_indices.add(i)
            word_matches.append({
                "word": expected_word,
                "isCorrect": True,
                "isInPosition": True,
                "userWord": user_word,
                "position": i,
                "suggestedWord": None
            })

    # Second pass: find remaining correct words
    for i in range(len(expected_words)):
        if any(m.get("position") == i and m.get("isInPosition") for m in word_matches):
            continue

        expected_word = expected_words[i]
        found = False

        for j in range(len(user_words)):
            if j in matched_indices:
                continue

            user_word = user_words[j]

            if (user_word == expected_word or
                    normalize_danish_word(user_word) == normalize_danish_word(expected_word)):
                matched_indices.add(j)
                correct_words += 1
                word_matches.append({
                    "word": expected_word,
                    "isCorrect": True,
                    "isInPosition": False,
                    "userWord": user_word,
                    "position": i,
                    "expectedPosition": i,
                    "actualPosition": j,
                    "suggestedWord": user_word
                })
                found = True
                break

        if not found:
            word_matches.append({
                "word": expected_word,
                "isCorrect": False,
                "isInPosition": False,
                "position": i,
                "suggestedWord": "?"
            })

    # Sort word matches by original position
    word_matches.sort(key=lambda x: x["position"])

    # Calculate accuracies
    word_accuracy = round((correct_words / len(expected_words)) * 100) if expected_words else 0
    position_accuracy = round((correct_positions / len(expected_words)) * 100) if expected_words else 0
    final_accuracy = round((word_accuracy * 0.7) + (position_accuracy * 0.3))

    # Generate feedback
    feedback = get_feedback_message(final_accuracy, correct_positions, len(expected_words))
    detailed_analysis = generate_detailed_analysis(final_accuracy, correct_words, correct_positions,
                                                   len(expected_words), word_matches)

    return {
        "accuracy": final_accuracy,
        "wordAccuracy": word_accuracy,
        "positionAccuracy": position_accuracy,
        "feedback": feedback,
        "wordMatches": word_matches,
        "detailedAnalysis": detailed_analysis
    }


def get_feedback_message(accuracy, correct_positions, total_words):
    """Generate appropriate feedback message based on accuracy"""
    if accuracy >= 95:
        return "Excellent! Totally perfect! 🌟"
    if accuracy >= 85:
        return "Great! Almost perfect! ✨"
    if accuracy >= 70:
        if correct_positions == total_words:
            return "Nice job! The words are in the right order! 👍"
        else:
            return "Nice job! Try to focus on word order 🎯"
    if accuracy >= 50:
        if correct_positions < total_words / 2:
            return "Not bad! Remember, word order is very important in danish 📝"
        else:
            return "Not bad! Keep practicing! 💪"
    if accuracy >= 30:
        return "Keep going! Try again 📚"
    if accuracy >= 10:
        return "Start slowly, say every word clearly 🗣️"
    return "Try again, take your time with each word 💪"


def generate_detailed_analysis(final_accuracy, correct_words, correct_positions, total_words, word_matches):
    """Generate detailed analysis of pronunciation"""
    if total_words == 0:
        return "No words to compare"

    incorrect_words = [w for w in word_matches if not w.get("isCorrect", False)]
    out_of_position_words = [w for w in word_matches if w.get("isCorrect", False) and not w.get("isInPosition", False)]

    if len(incorrect_words) == 0 and len(out_of_position_words) == 0:
        return f"Perfect! All {total_words} words are pronunced correctly and in the right order! 🎉"

    if len(incorrect_words) == 0 and len(out_of_position_words) > 0:
        if len(out_of_position_words) == 1:
            return f"You pronunced every word correctly, but 1 word is in the wrong place: '{out_of_position_words[0]['word']}' should be at position {out_of_position_words[0]['position'] + 1}. Focus on word order! 📝"
        else:
            words_str = ", ".join([f"'{w['word']}'" for w in out_of_position_words])
            return f"You pronunced every word correctly, but {len(out_of_position_words)} words are in the wrong place: {words_str}. Focus on word order! 📝"

    if len(incorrect_words) == 1:
        return f"Almost perfect! Only one word to work on: '{incorrect_words[0]['word']}'"

    problem_words = ", ".join([f"'{w['word']}'" for w in incorrect_words[:3]])
    if len(incorrect_words) > 3:
        return f"You got {correct_words} out of {total_words} words correct. {correct_positions} of them were in the right place. Focus on: {problem_words} and {len(incorrect_words) - 3} more"

    return f"You got {correct_words} out of {total_words} words correctly. {correct_positions} of them were in the right place. Focus on: {problem_words}"


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

        # Transcribe
        transcript = asr_model.transcribe([temp_path])

        # Clean up temp file
        os.unlink(temp_path)

        return transcript[0].text
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
            flashcards = get_flashcard_collection()
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
        flashcards = get_flashcard_collection()

        # Apply pagination
        total = len(flashcards)
        paginated = flashcards[offset:offset + limit]

        # Log user activity
        user = User.find_by_id(flask.g.user_id)
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
        flashcards = get_flashcard_collection()
        flashcard = next((f for f in flashcards if f['id'] == flashcard_id), None)

        if not flashcard:
            return json_result({"error": "Flashcard not found"}), 404

        # Log user activity
        user = User.find_by_id(flask.g.user_id)
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

        # For now, return random cards as a mock
        flashcards = get_flashcard_collection()

        # Shuffle and take first 'count' cards
        random.shuffle(flashcards)
        practice_cards = flashcards[:count]

        # Add user-specific data (in real implementation, this would come from database)
        for card in practice_cards:
            card["user_progress"] = {
                "repetitions": random.randint(0, 5),
                "last_practiced": None,
                "ease_factor": 2.5,
                "interval": 0
            }

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
        "response_time_ms": 5000
    }
    
    Returns updated user progress and accuracy analysis.
    """
    try:
        data = request.get_json()
        if not data:
            return json_result({"error": "JSON body required"}), 400

        flashcard_id = data.get('flashcard_id')
        user_answer = data.get('user_answer', '')
        is_correct = data.get('is_correct')
        answer_source = data.get('answer_source', 'unknown')
        response_time = data.get('response_time_ms', 0)

        if not flashcard_id or is_correct is None:
            return json_result({"error": "flashcard_id and is_correct are required"}), 400

        # Get the flashcard
        flashcards = get_flashcard_collection()
        flashcard = next((f for f in flashcards if f['id'] == flashcard_id), None)

        if not flashcard:
            return json_result({"error": "Flashcard not found"}), 404

        user = User.find_by_id(flask.g.user_id)

        # Calculate accuracy analysis if user_answer is provided
        accuracy_analysis = None
        if user_answer and not is_correct:  # Calculate even for correct answers to provide feedback
            expected_text = flashcard.get('expectedText', flashcard['prompt'])
            accuracy_analysis = calculate_accuracy(user_answer, expected_text)

            # Override is_correct based on accuracy if needed
            if accuracy_analysis['accuracy'] >= 70:
                is_correct = True

        # Mock response with updated progress
        new_interval = random.choice([1, 2, 4, 7, 14, 30]) if is_correct else 1

        log(f"User {user.id} answered flashcard {flashcard_id}: correct={is_correct}, source={answer_source}, time={response_time}ms, answer='{user_answer}'")

        response_data = {
            "success": True,
            "flashcard_id": flashcard_id,
            "is_correct": is_correct,
            "next_review_days": new_interval,
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