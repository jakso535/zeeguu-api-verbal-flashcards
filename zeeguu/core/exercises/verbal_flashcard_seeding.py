from datetime import datetime

from sqlalchemy.orm import joinedload

from zeeguu.core.model.bookmark import Bookmark
from zeeguu.core.model.bookmark_user_preference import UserWordExPreference
from zeeguu.core.model.context_identifier import ContextIdentifier
from zeeguu.core.model.context_type import ContextType
from zeeguu.core.model.meaning import Meaning
from zeeguu.core.model.user_word import UserWord
from zeeguu.core.word_scheduling.basicSR.four_levels_per_word import FourLevelsPerWord


VERBAL_FLASHCARD_TEST_WORDS = [
    ("hus", "house"),
    ("bil", "car"),
    ("cykel", "bicycle"),
    ("tog", "train"),
    ("bus", "bus"),
    ("fly", "airplane"),
    ("båd", "boat"),
    ("vej", "road"),
    ("gade", "street"),
    ("bro", "bridge"),
    ("by", "city"),
    ("landsby", "village"),
    ("skole", "school"),
    ("bog", "book"),
    ("blyant", "pencil"),
    ("bord", "table"),
    ("stol", "chair"),
    ("dør", "door"),
    ("vindue", "window"),
    ("nøgle", "key"),
    ("have", "garden"),
    ("park", "park"),
    ("strand", "beach"),
    ("skov", "forest"),
    ("bjerg", "mountain"),
    ("sø", "lake"),
    ("flod", "river"),
    ("hav", "sea"),
    ("sol", "sun"),
    ("måne", "moon"),
    ("stjerne", "star"),
    ("regn", "rain"),
    ("sne", "snow"),
    ("vind", "wind"),
    ("sommer", "summer"),
    ("vinter", "winter"),
    ("forår", "spring"),
    ("efterår", "autumn"),
    ("hund", "dog"),
    ("kat", "cat"),
    ("fugl", "bird"),
    ("fisk", "fish"),
    ("hest", "horse"),
    ("ko", "cow"),
    ("gris", "pig"),
    ("barn", "child"),
    ("ven", "friend"),
    ("familie", "family"),
    ("mor", "mother"),
    ("far", "father"),
    ("søster", "sister"),
    ("bror", "brother"),
    ("baby", "baby"),
    ("lærer", "teacher"),
    ("elev", "student"),
    ("arbejde", "work"),
    ("kontor", "office"),
    ("butik", "shop"),
    ("penge", "money"),
    ("mad", "food"),
    ("brød", "bread"),
    ("smør", "butter"),
    ("ost", "cheese"),
    ("mælk", "milk"),
    ("kaffe", "coffee"),
    ("te", "tea"),
    ("vand", "water"),
    ("juice", "juice"),
    ("æble", "apple"),
    ("banan", "banana"),
    ("appelsin", "orange"),
    ("kartoffel", "potato"),
    ("tomat", "tomato"),
    ("blomst", "flower"),
    ("træ", "tree"),
    ("blad", "leaf"),
    ("græs", "grass"),
    ("bold", "ball"),
    ("fodbold", "football"),
    ("musik", "music"),
    ("film", "movie"),
    ("billede", "picture"),
    ("telefon", "telephone"),
    ("computer", "computer"),
    ("ur", "clock"),
    ("taske", "bag"),
    ("jakke", "jacket"),
    ("sko", "shoe"),
    ("hat", "hat"),
    ("seng", "bed"),
    ("lampe", "lamp"),
    ("køkken", "kitchen"),
    ("bad", "bathroom"),
    ("gulv", "floor"),
    ("væg", "wall"),
    ("tag", "roof"),
    ("posthus", "post office"),
    ("hospital", "hospital"),
    ("marked", "market"),
    ("museum", "museum"),
    ("bibliotek", "library"),
]


def _pair_key(origin, translation):
    return (origin.casefold().strip(), translation.casefold().strip())


def _ensure_schedule(session, user_word):
    schedule = FourLevelsPerWord.find(user_word)
    if schedule:
        return schedule

    schedule = FourLevelsPerWord(user_word=user_word)
    session.add(schedule)
    return schedule


def _activate_user_word(session, user_word):
    user_word.user_preference = UserWordExPreference.USE_IN_EXERCISES
    user_word.fit_for_study = True
    user_word.learned_time = None
    user_word.level = 3
    user_word.is_user_added = True

    if user_word.preferred_bookmark is None:
        bookmarks = user_word.bookmarks()
        if bookmarks:
            user_word.preferred_bookmark = bookmarks[0]

    schedule = _ensure_schedule(session, user_word)
    schedule.next_practice_time = datetime.now()
    schedule.cooling_interval = 0
    schedule.consecutive_correct_answers = 0

    session.add(user_word)
    session.add(schedule)
    return schedule


def _seed_pair_for_user(session, user, origin, translation):
    meaning = Meaning.find_or_create(session, origin, "da", translation, "en")
    meaning.validated = Meaning.VALID
    session.add(meaning)

    user_word = UserWord.find_or_create(session, user, meaning, is_user_added=True)
    user_word.is_user_added = True

    context_type = ContextType.find_or_create(
        session, ContextType.USER_EDITED_TEXT, commit=False
    )
    session.add(context_type)

    bookmark = Bookmark.find_or_create(
        session,
        user,
        origin,
        "da",
        translation,
        "en",
        origin,
        None,
        None,
        sentence_i=0,
        token_i=0,
        total_tokens=1,
        c_sentence_i=0,
        c_token_i=0,
        context_identifier=ContextIdentifier(ContextType.USER_EDITED_TEXT),
    )

    user_word.preferred_bookmark = bookmark
    _activate_user_word(session, user_word)
    return user_word


def _load_existing_seed_words(user):
    return (
        UserWord.query.filter(UserWord.user_id == user.id)
        .options(
            joinedload(UserWord.meaning).joinedload(Meaning.origin),
            joinedload(UserWord.meaning).joinedload(Meaning.translation),
            joinedload(UserWord.preferred_bookmark),
        )
        .all()
    )


def seed_verbal_flashcards_for_user(session, user, count=20):
    existing_words = _load_existing_seed_words(user)
    existing_by_key = {
        _pair_key(
            user_word.meaning.origin.content,
            user_word.meaning.translation.content,
        ): user_word
        for user_word in existing_words
        if user_word.meaning and user_word.meaning.origin and user_word.meaning.translation
    }

    seeded_pairs = []
    refreshed_pairs = []

    available_pairs = [
        pair
        for pair in VERBAL_FLASHCARD_TEST_WORDS
        if _pair_key(*pair) not in existing_by_key
    ]

    for origin, translation in available_pairs[:count]:
        _seed_pair_for_user(session, user, origin, translation)
        seeded_pairs.append({"origin": origin, "translation": translation})

    remaining = count - len(seeded_pairs)
    if remaining > 0:
        for origin, translation in VERBAL_FLASHCARD_TEST_WORDS:
            if remaining == 0:
                break

            key = _pair_key(origin, translation)
            user_word = existing_by_key.get(key)
            if not user_word:
                continue

            _activate_user_word(session, user_word)
            refreshed_pairs.append({"origin": origin, "translation": translation})
            remaining -= 1

    session.commit()

    return {
        "requested_count": count,
        "seeded_count": len(seeded_pairs),
        "refreshed_count": len(refreshed_pairs),
        "seeded_words": seeded_pairs,
        "refreshed_words": refreshed_pairs,
        "total_selected": len(seeded_pairs) + len(refreshed_pairs),
    }
