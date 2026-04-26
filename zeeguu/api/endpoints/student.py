import flask
from flask import request, jsonify
from zeeguu.api.utils import json_result

from zeeguu.core.model import Cohort, User
from zeeguu.core.model.user_cohort_map import UserCohortMap

from zeeguu.api.utils.route_wrappers import cross_domain, requires_session
from . import api, db_session


# ---------------------------------------------------------------------------
@api.route("/join_cohort", methods=("POST",))
# ---------------------------------------------------------------------------
@cross_domain
@requires_session
def join_cohort_api():
    invite_code = request.form.get("invite_code", "").strip()

    if not invite_code:
        return jsonify({"status": "error", "message": "Missing invite code"}), 400

    # Validate invite code
    try:
        cohort = Cohort.find_by_code(invite_code)
    except:
        return jsonify({"status": "error", "message": "Invalid invite code"}), 400

    user = User.find_by_id(flask.g.user_id)

    # Idempotent join: if already in cohort, return success
    existing = UserCohortMap.query.filter_by(
        user_id=user.id,
        cohort_id=cohort.id
    ).first()

    if not existing:
        db_session.add(UserCohortMap(user_id=user.id, cohort_id=cohort.id))
        db_session.commit()

    return jsonify({"status": "ok", "cohort_id": cohort.id}), 200


# ---------------------------------------------------------------------------
@api.route("/student_info", methods=["GET"])
# ---------------------------------------------------------------------------
@cross_domain
@requires_session
def student_info():
    user = User.find_by_id(flask.g.user_id)
    user_cohorts = [c.cohort.get_cohort_info() for c in user.cohorts]
    return json_result(
        {
            "name": user.name,
            "email": user.email,
            "cohorts": user_cohorts,
        }
    )


# ---------------------------------------------------------------------------
@api.route("/cohort_name/<id>", methods=["GET"])
# ---------------------------------------------------------------------------
@requires_session
def cohort_name(id):

    cohort = Cohort.find(id)
    return {"name": cohort.name}
