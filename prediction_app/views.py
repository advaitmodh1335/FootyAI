# prediction_app/views.py
from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Optional, Dict

import joblib
import pandas as pd
from django.conf import settings
from django.db.models import Avg, Count
from django.http import HttpRequest, JsonResponse
from django.views.decorators.http import require_GET

from .models import Match

# ============================================================
# === Model & Encoder Paths ==================================
# ============================================================
_MODEL = None
_ENCODER = None

_BASE_DIR = Path(settings.BASE_DIR)
_MODEL_PATH = _BASE_DIR / "soccer_predictor_model.joblib"
_CALIBRATED_PATH = _BASE_DIR / "soccer_predictor_calibrated.joblib"
_ENCODER_PATH = _BASE_DIR / "label_encoder.joblib"


def _get_model():
    """
    Lazy-load the model, preferring the calibrated version if available.
    """
    global _MODEL
    if _MODEL is None:
        path = _CALIBRATED_PATH if _CALIBRATED_PATH.exists() else _MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(f"Model not found at {path}")
        _MODEL = joblib.load(path)
    return _MODEL


def _get_encoder():
    """
    Lazy-load the LabelEncoder used to create team_code and opp_code.
    """
    global _ENCODER
    if _ENCODER is None:
        if not _ENCODER_PATH.exists():
            raise FileNotFoundError(f"Label encoder not found at {_ENCODER_PATH}")
        _ENCODER = joblib.load(_ENCODER_PATH)
    return _ENCODER


# ============================================================
# === Utilities ==============================================
# ============================================================
def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _canonicalize(s: str) -> str:
    return _strip_accents(s).casefold().strip()


def _all_team_names():
    """
    Return all unique team names seen as team or opponent.
    """
    home = Match.objects.values_list("team", flat=True).distinct()
    away = Match.objects.values_list("opponent", flat=True).distinct()
    return sorted(set(home).union(set(away)))


def _resolve_team_name(user_input: Optional[str]) -> Optional[str]:
    """
    Map user input (case/accents tolerant) to the canonical DB name.
    """
    if not user_input:
        return None
    can = _canonicalize(user_input)
    for name in _all_team_names():
        if _canonicalize(name) == can:
            return name
    return None


# ============================================================
# === Feature Builder (MUST match train_model.py) ============
# ============================================================
def _team_recent_form(team: str) -> Dict[str, float]:
    """
    Compute last-5 average for key stats used in training (gf, ga, sh, sot).
    """
    qs = Match.objects.filter(team=team).order_by("-date")[:5]
    if not qs.exists():
        return {"gf": 0.0, "ga": 0.0, "sh": 0.0, "sot": 0.0}

    agg = qs.aggregate(
        gf=Avg("gf"),
        ga=Avg("ga"),
        sh=Avg("sh"),
        sot=Avg("sot"),
    )
    return {
        "gf": float(agg.get("gf") or 0.0),
        "ga": float(agg.get("ga") or 0.0),
        "sh": float(agg.get("sh") or 0.0),
        "sot": float(agg.get("sot") or 0.0),
    }


def _build_feature_row(home: str, away: str) -> pd.DataFrame:
    """
    Construct a single-row DataFrame with columns:
    [
      'venue_code','team_code','opp_code',
      'gf_rolling','ga_rolling','sh_rolling','sot_rolling',
      'opp_gf_rolling','opp_ga_rolling','opp_sh_rolling','opp_sot_rolling'
    ]
    """
    # Ensure artifacts exist
    _get_model()
    le = _get_encoder()

    # Encode teams (fallback -1 for unseen to avoid exceptions)
    try:
        home_code = int(le.transform([home])[0])
    except Exception:
        home_code = -1
    try:
        away_code = int(le.transform([away])[0])
    except Exception:
        away_code = -1

    # Your training uses a numeric venue_code; at inference we predict for the home side.
    venue_code = 1

    home_form = _team_recent_form(home)
    away_form = _team_recent_form(away)

    row = {
        "venue_code": venue_code,
        "team_code": home_code,
        "opp_code": away_code,
        "gf_rolling": home_form["gf"],
        "ga_rolling": home_form["ga"],
        "sh_rolling": home_form["sh"],
        "sot_rolling": home_form["sot"],
        "opp_gf_rolling": away_form["gf"],
        "opp_ga_rolling": away_form["ga"],
        "opp_sh_rolling": away_form["sh"],
        "opp_sot_rolling": away_form["sot"],
    }
    # Preserve column order explicitly
    cols = list(row.keys())
    return pd.DataFrame([row], columns=cols)


# ============================================================
# === API Endpoints ==========================================
# ============================================================
@require_GET
def health(request: HttpRequest):
    """
    Health probe + quick facts to help debugging.
    """
    try:
        payload = {
            "ok": True,
            "db_has_data": Match.objects.exists(),
            "model_exists": _MODEL_PATH.exists() or _CALIBRATED_PATH.exists(),
            "calibrated_present": _CALIBRATED_PATH.exists(),
            "encoder_exists": _ENCODER_PATH.exists(),
        }
        return JsonResponse(payload, status=200)
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=500)


@require_GET
def get_teams_and_matches(request: HttpRequest):
    """
    Returns the unique list of team names for dropdowns.
    """
    try:
        return JsonResponse({"teams": _all_team_names()}, status=200)
    except Exception as e:
        return JsonResponse({"error": f"Failed to fetch teams: {e}"}, status=500)


@require_GET
def team_data(request: HttpRequest, team: Optional[str] = None):
    """
    Supports both:
      /api/team_data/?team=Barcelona
      /api/team_data/Barcelona/
    Returns:
      {
        "team": "...",
        "samples": <int>,
        "summary": {
          "avg_gf": <float>, "avg_ga": <float>, "avg_sh": <float>, "avg_sot": <float>,
          "record": "W-L-D"
        },
        "history": [ { date, opponent, venue, result, gf, ga, formation, captain, referee } ... ]
      }
    """
    team = team or request.GET.get("team")
    if not team:
        return JsonResponse({"error": "Missing 'team' parameter"}, status=400)

    resolved = _resolve_team_name(team) or team
    qs = Match.objects.filter(team=resolved).order_by("-date")

    if not qs.exists():
        return JsonResponse(
            {"team": resolved, "samples": 0, "summary": {}, "history": []},
            status=200,
        )

    # Summary stats
    summary = qs.aggregate(
        matches=Count("id"),
        avg_gf=Avg("gf"),
        avg_ga=Avg("ga"),
        avg_sh=Avg("sh"),
        avg_sot=Avg("sot"),
    )

    # Record W-L-D
    wins = qs.filter(result__iexact="W").count()
    draws = qs.filter(result__iexact="D").count()
    losses = qs.filter(result__iexact="L").count()
    summary["record"] = f"{wins}-{losses}-{draws}"

    # Recent match history (latest 10)
    history = list(
        qs.values(
            "date",
            "opponent",
            "venue",
            "result",
            "gf",
            "ga",
            "formation",
            "captain",
            "referee",
        )[:10]
    )
    # Normalize fields for robustness
    for m in history:
        m["date"] = str(m.get("date") or "")
        m["venue"] = m.get("venue") or "Unknown"
        m["formation"] = m.get("formation") or "-"
        m["captain"] = m.get("captain") or "-"
        m["referee"] = m.get("referee") or "-"

    return JsonResponse(
        {
            "team": resolved,
            "samples": int(summary.pop("matches", 0) or 0),
            "summary": summary,
            "history": history,
        },
        status=200,
    )


@require_GET
def predict(request: HttpRequest):
    """
    Predict endpoint:
      /api/predict/?home_team=Barcelona&away_team=Real%20Madrid
    Also accepts ?home=&away= as fallbacks.
    Returns:
      {
        "home": "...", "away": "...",
        "prediction": 0|1,
        "prob_home_win": <float or null>,
        "features_used": { ... }
      }
    """
    home_in = request.GET.get("home_team") or request.GET.get("home")
    away_in = request.GET.get("away_team") or request.GET.get("away")

    if not home_in or not away_in:
        return JsonResponse({"error": "Missing home_team or away_team parameter"}, status=400)
    if home_in == away_in:
        return JsonResponse({"error": "Home and away teams cannot be the same"}, status=400)

    home = _resolve_team_name(home_in) or home_in
    away = _resolve_team_name(away_in) or away_in

    if not Match.objects.filter(team=home).exists():
        return JsonResponse({"error": f"No data found for team: {home}"}, status=404)
    if not Match.objects.filter(team=away).exists():
        return JsonResponse({"error": f"No data found for team: {away}"}, status=404)

    try:
        model = _get_model()
        X = _build_feature_row(home, away)

        # Probability (if supported)
        prob = None
        try:
            prob = float(model.predict_proba(X)[0][1])
        except Exception:
            pass

        pred = int(model.predict(X)[0])  # 1 = home win, 0 = not home win

        return JsonResponse(
            {
                "home": home,
                "away": away,
                "prediction": pred,
                "prob_home_win": prob,
                "features_used": X.to_dict(orient="records")[0],
            },
            status=200,
        )
    except FileNotFoundError as e:
        return JsonResponse({"error": str(e)}, status=500)
    except Exception as e:
        return JsonResponse({"error": f"Prediction failed: {e}"}, status=500)


# --- Optional legacy alias so old routes don't break ---
@require_GET
def predict_match(request: HttpRequest):
    """
    Legacy alias:
      /api/predict_match/?home=...&away=...
    Delegates to the same logic as /api/predict/.
    """
    # Proxy params into the predict() handler
    return predict(request)