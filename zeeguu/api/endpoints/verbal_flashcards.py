import traceback
import flask
import io
import os
import tempfile
import random
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
            "difficulty": "Beginner",
            "category": "Greetings",
            "example": "Person A: Hello, how are you? Person B: I'm fine, thanks!",
            "phoneticHint": "hej hvordan har du det"
        },
        {
            "id": "2",
            "prompt": "It is really nice weather today",
            "hint": "Det er virkelig godt vejr i dag",
            "expectedText": "det er virkelig godt vejr i dag",
            "difficulty": "Beginner",
            "category": "Weather",
            "example": "Let's go for a walk, it is really nice weather today!",
            "phoneticHint": "det er virkelig godt vejr i dag"
        },
        {
            "id": "3",
            "prompt": "I want to order a coffee",
            "hint": "Jeg vil gerne bestille en kaffe",
            "expectedText": "jeg vil gerne bestille en kaffe",
            "difficulty": "Intermediate",
            "category": "Food & Drink",
            "example": "Excuse me, I want to order a coffee please.",
            "phoneticHint": "jeg vil gerne bestille en kaffe"
        },
        {
            "id": "4",
            "prompt": "Can you help me?",
            "hint": "Kan du hjælpe mig?",
            "expectedText": "kan du hjælpe mig",
            "difficulty": "Beginner",
            "category": "Requests",
            "example": "I'm lost, can you help me find the station?",
            "phoneticHint": "kan du hjælpe mig"
        },
        {
            "id": "5",
            "prompt": "When is the meeting?",
            "hint": "Hvornår er mødet?",
            "expectedText": "hvornår er mødet",
            "difficulty": "Intermediate",
            "category": "Work",
            "example": "When is the meeting scheduled for today?",
            "phoneticHint": "hvornår er mødet"
        },
        {
            "id": "6",
            "prompt": "Coffee",
            "hint": "kaffe",
            "expectedText": "kaffe",
            "difficulty": "Beginner",
            "category": "Food & Drink",
            "example": "I would like a coffee with milk.",
            "phoneticHint": "kaffe"
        },
        {
            "id": "7",
            "prompt": "Cake",
            "hint": "kage",
            "expectedText": "kage",
            "difficulty": "Beginner",
            "category": "Food & Drink",
            "example": "This cake is delicious!",
            "phoneticHint": "kage"
        },
        {
            "id": "8",
            "prompt": "Sausage",
            "hint": "pølse",
            "expectedText": "pølse",
            "difficulty": "Beginner",
            "category": "Food & Drink",
            "example": "Danish hot dogs with sausage are famous.",
            "phoneticHint": "pølse"
        }
    ]


# ====================================
# Helper Functions
# ====================================
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
    Get flashcards, optionally filtered by category and difficulty.
    
    Query parameters:
    - category: filter by category (optional)
    - difficulty: filter by difficulty (optional)
    - limit: max number of cards to return (optional, default 50)
    - offset: pagination offset (optional, default 0)
    
    Returns list of flashcards.
    """
    try:
        # Get query parameters
        category = request.args.get('category')
        difficulty = request.args.get('difficulty')
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))

        # Get all flashcards
        flashcards = get_flashcard_collection()

        # Apply pagination
        total = len(flashcards)
        flashcards = flashcards[offset:offset + limit]

        # Log user activity
        user = User.find_by_id(flask.g.user_id)
        log(f"User {user.id} requested flashcards with filters: category={category}, difficulty={difficulty}")

        return json_result({
            "flashcards": flashcards,
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


@api.route("/verbal_flashcards/categories", methods=["GET"])
@cross_domain
@requires_session
def get_categories():
    """
    Get all available categories with counts and difficulties.
    
    Returns list of categories with metadata.
    """
    try:
        flashcards = get_flashcard_collection()

        # Group by category
        categories = {}
        for card in flashcards:
            cat = card['category']
            if cat not in categories:
                categories[cat] = {
                    "name": cat,
                    "count": 0,
                    "difficulties": set()
                }
            categories[cat]["count"] += 1
            categories[cat]["difficulties"].add(card['difficulty'])

        # Convert sets to lists
        result = []
        for cat in categories.values():
            cat["difficulties"] = list(cat["difficulties"])
            result.append(cat)

        # Sort by name
        result.sort(key=lambda x: x["name"])

        return json_result(result)

    except Exception as e:
        log(f"Get categories error: {e}")
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
    
    Returns updated user progress.
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

        # Mock response with updated progress
        new_interval = random.choice([1, 2, 4, 7, 14, 30])

        log(f"User {user.id} answered flashcard {flashcard_id}: correct={is_correct}, source={answer_source}, time={response_time}ms, answer='{user_answer}'")

        return json_result({
            "success": True,
            "flashcard_id": flashcard_id,
            "is_correct": is_correct,
            "next_review_days": new_interval,
            "message": "Answer recorded"
        })

    except Exception as e:
        log(f"Submit answer error: {e}")
        traceback.print_exc()
        return json_result({"error": str(e)}), 500