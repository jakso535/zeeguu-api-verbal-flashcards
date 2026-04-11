import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from zeeguu.api.app import create_app
from zeeguu.core.exercises.verbal_flashcard_seeding import (
    seed_verbal_flashcards_for_user,
)
from zeeguu.core.model.db import db
from zeeguu.core.model.user import User


def parse_args():
    parser = argparse.ArgumentParser(
        description="Seed simple Danish verbal flashcard words for a user."
    )
    parser.add_argument("--email", required=True, help="User email to seed")
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="How many words to add or refresh (default: 20)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    app = create_app()

    with app.app_context():
        try:
            user = User.find(args.email)
        except Exception:
            print(f"User not found: {args.email}", file=sys.stderr)
            return 1

        result = seed_verbal_flashcards_for_user(db.session, user, count=args.count)

        print(
            f"Seeded verbal flashcards for {user.email}: "
            f"{result['seeded_count']} new, {result['refreshed_count']} refreshed"
        )

        if result["seeded_words"]:
            print("New words:")
            for word in result["seeded_words"]:
                print(f"  - {word['origin']} -> {word['translation']}")

        if result["refreshed_words"]:
            print("Refreshed words:")
            for word in result["refreshed_words"]:
                print(f"  - {word['origin']} -> {word['translation']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
