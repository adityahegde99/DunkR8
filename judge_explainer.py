"""
Natural-language dunk reasoning via DeepSeek (with safe local fallback).
"""
from __future__ import annotations

from typing import Any, Dict, Optional
import json
import urllib.request
import urllib.error


DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"


def _fallback_reasoning(analysis: Any) -> str:
    if not analysis.is_dunk:
        return (
            f"I'm ruling this attempt as incomplete. {analysis.rejection_reason}. "
            f"While there were some athletic elements - {analysis.max_vertical_inches:.0f} inches of elevation "
            f"and {analysis.hang_time_s:.2f}s hang time - the execution didn't meet contest standards. "
            f"Final ruling: {analysis.final_contest_score:.1f}/50."
        )
    
    # For valid dunks, give judge-style breakdown
    comps = analysis.score_components
    dunk_name = analysis.dunk_type.replace(" Dunk", "").replace("Standard ", "")
    
    strengths = []
    if comps.judge_athleticism >= 9.0:
        strengths.append(f"exceptional athleticism with {analysis.hang_time_s:.2f}s of hang time")
    elif analysis.max_vertical_inches >= 30:
        strengths.append(f"solid {analysis.max_vertical_inches:.0f}-inch elevation")
    
    if comps.judge_difficulty >= 9.0:
        strengths.append("high degree of difficulty")
    elif analysis.rotation_degrees >= 180:
        strengths.append(f"{analysis.rotation_degrees:.0f}° of body rotation")
        
    if analysis.alley_oop or analysis.self_lob:
        strengths.append("well-executed lob timing")
    if analysis.over_object:
        strengths.append("clearing an obstacle")
        
    strength_text = ", ".join(strengths) if strengths else "solid fundamentals"
    
    return (
        f"This {dunk_name} showcases {strength_text}. "
        f"The execution was {'clean' if comps.judge_execution >= 9.0 else 'solid'} "
        f"with {'great' if comps.judge_creativity >= 9.0 else 'good'} presentation. "
        f"Final score: {analysis.final_contest_score:.1f}/50."
    )


def _build_prompt_payload(analysis: Any) -> Dict[str, Any]:
    evidence: Dict[str, Any] = {
        "is_dunk": analysis.is_dunk,
        "dunk_type": analysis.dunk_type,
        "primary_category": analysis.primary_category,
        "non_dunk_type": analysis.non_dunk_type,
        "rejection_reason": analysis.rejection_reason,
        "hang_time_s": round(float(analysis.hang_time_s), 3),
        "max_vertical_inches": round(float(analysis.max_vertical_inches), 2),
        "rotation_degrees": round(float(analysis.rotation_degrees), 1),
        "ball_air_time_s": round(float(analysis.ball_air_time_s), 3),
        "takeoff_distance_ft": round(float(analysis.takeoff_distance_ft), 2),
        "alley_oop": bool(analysis.alley_oop),
        "self_lob": bool(analysis.self_lob),
        "over_object": bool(analysis.over_object),
        "lob_type": getattr(analysis, 'lob_type', 'none'),
        "model_prediction": analysis.model_prediction,
        "model_confidence": round(float(analysis.model_confidence), 3),
        "dunk_probability": round(float(analysis.dunk_probability), 3),
        "dunk_type_confidence": round(float(analysis.dunk_type_confidence), 3),
        "score_confidence": round(float(analysis.score_confidence), 3),
        "final_contest_score": round(float(analysis.final_contest_score), 1),
        "judge_scores": {
            "difficulty": round(float(analysis.score_components.judge_difficulty), 1),
            "execution": round(float(analysis.score_components.judge_execution), 1),
            "creativity": round(float(analysis.score_components.judge_creativity), 1),
            "athleticism": round(float(analysis.score_components.judge_athleticism), 1),
            "style": round(float(analysis.score_components.judge_style), 1),
        }
    }
    
    system = (
        "You are an NBA slam dunk contest judge with years of experience. Follow official NBA judging criteria:\n\n"
        "**Official NBA Judging Criteria (40-50 point scale):**\n"
        "• Creativity and originality - Innovation, uniqueness of the dunk\n"
        "• Degree of difficulty - Technical complexity, risk factor\n" 
        "• Clean execution - Control, smooth landing, ball handling\n"
        "• Athleticism - Hang time, vertical leap, body control\n"
        "• Crowd and visual impact - Entertainment value, style, flair\n\n"
        "**Your Role:**\n"
        "- Write as a seasoned NBA judge (like Dominique Wilkins or Dwight Howard)\n"
        "- Use basketball terminology naturally (elevation, rotation, extension, etc.)\n"
        "- Reference specific technical aspects that influenced your scoring\n"
        "- Be decisive but fair in your assessment\n"
        "- Keep it conversational but professional (3-4 sentences)\n"
        "- NEVER mention AI, algorithms, or technical analysis tools\n"
        "- Do not use em dashes (—). Use commas or ' - ' instead.\n"
        "- Judge the dunk as if you saw it live at All-Star Weekend"
    )
    
    user = (
        f"Judge this dunk attempt based on the performance data:\n\n"
        f"**Dunk Analysis:**\n"
        f"{json.dumps(evidence, indent=2)}\n\n"
        f"**Your Task:**\n"
        f"Provide your judge's commentary explaining the {evidence['final_contest_score']}/50 score. "
        f"Reference the specific aspects that stood out - both strengths and any areas that held the score back. "
        f"Write as if explaining your scoring to TV audience during the contest.\n\n"
        f"Output only your judge commentary (no JSON, no technical details)."
    )
    
    return {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.7,
        "max_tokens": 300,
    }


def get_ai_score_adjustment(
    analysis: Any,
    api_key: Optional[str],
    timeout_s: float = 8.0,
) -> float:
    """
    Get AI judge's score adjustment based on dunk type and complexity.
    Returns adjustment factor between -2.0 and +2.0 to modify final score.
    """
    key = (api_key or "").strip()
    if not key or not analysis.is_dunk:
        return 0.0

    evidence = {
        "dunk_type": analysis.dunk_type,
        "primary_category": analysis.primary_category,
        "hang_time_s": round(float(analysis.hang_time_s), 3),
        "max_vertical_inches": round(float(analysis.max_vertical_inches), 2),
        "rotation_degrees": round(float(analysis.rotation_degrees), 1),
        "alley_oop": bool(analysis.alley_oop),
        "self_lob": bool(analysis.self_lob),
        "over_object": bool(analysis.over_object),
        "lob_type": getattr(analysis, 'lob_type', 'none'),
        "model_confidence": round(float(analysis.model_confidence), 3),
        "current_score": round(float(analysis.final_contest_score), 1),
    }

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an experienced NBA dunk contest judge reviewing a score. "
                    "Based on dunk type complexity and execution factors, suggest a score adjustment.\n\n"
                    "**Scoring Guidelines:**\n"
                    "- Basic dunks (Power, One-Hand): baseline scoring\n"
                    "- Technical dunks (Windmill, 360, Reverse): +0.5 to +1.5 bonus\n"
                    "- Elite dunks (Double Pump, Between-Legs, 540): +1.0 to +2.0 bonus\n"
                    "- Lob timing adds: +0.3 to +0.8\n"
                    "- Over obstacle adds: +0.5 to +1.0\n"
                    "- Poor execution reduces: -0.5 to -2.0\n\n"
                    "Return ONLY a number between -2.0 and +2.0 (the adjustment value)."
                )
            },
            {
                "role": "user", 
                "content": f"Dunk analysis: {json.dumps(evidence, indent=2)}\n\nWhat score adjustment do you recommend? (number only, -2.0 to +2.0)"
            }
        ],
        "temperature": 0.3,
        "max_tokens": 20,
    }

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            DEEPSEEK_URL,
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        parsed = json.loads(body)
        content = (
            parsed.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        # Parse the numeric response
        try:
            adjustment = float(content)
            return max(-2.0, min(2.0, adjustment))  # Clamp to valid range
        except ValueError:
            return 0.0
    except Exception:
        return 0.0


def get_ai_detection_assist(
    evidence: Dict[str, Any],
    api_key: Optional[str],
    timeout_s: float = 8.0,
) -> Dict[str, Any]:
    """
    Ask AI judge for borderline dunk-call assistance.
    Returns: {"is_dunk": bool, "dunk_type": str, "confidence": float}
    """
    key = (api_key or "").strip()
    if not key:
        return {"is_dunk": False, "dunk_type": "", "confidence": 0.0}

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an NBA dunk judge helping with borderline dunk detection. "
                    "Decide if this is a dunk and provide the most likely dunk type. "
                    "Return STRICT JSON only with keys: is_dunk (bool), dunk_type (string), confidence (0..1). "
                    "No extra text."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(evidence, indent=2),
            },
        ],
        "temperature": 0.2,
        "max_tokens": 120,
    }

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            DEEPSEEK_URL,
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        parsed = json.loads(body)
        content = (
            parsed.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        if not content:
            return {"is_dunk": False, "dunk_type": "", "confidence": 0.0}
        # Try direct JSON first
        try:
            obj = json.loads(content)
        except ValueError:
            # Fallback: extract first JSON object
            start = content.find("{")
            end = content.rfind("}")
            if start < 0 or end <= start:
                return {"is_dunk": False, "dunk_type": "", "confidence": 0.0}
            obj = json.loads(content[start : end + 1])
        return {
            "is_dunk": bool(obj.get("is_dunk", False)),
            "dunk_type": str(obj.get("dunk_type", "") or ""),
            "confidence": max(0.0, min(1.0, float(obj.get("confidence", 0.0) or 0.0))),
        }
    except Exception:
        return {"is_dunk": False, "dunk_type": "", "confidence": 0.0}


def generate_judge_reasoning(
    analysis: Any,
    api_key: Optional[str],
    timeout_s: float = 10.0,
) -> str:
    """
    Generate natural-language judging rationale.
    Falls back to deterministic local explanation when API unavailable.
    """
    key = (api_key or "").strip()
    if not key:
        return _fallback_reasoning(analysis)

    payload = _build_prompt_payload(analysis)
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        DEEPSEEK_URL,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        parsed = json.loads(body)
        content = (
            parsed.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        return content or _fallback_reasoning(analysis)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, KeyError, IndexError):
        return _fallback_reasoning(analysis)

