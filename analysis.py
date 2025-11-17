# analysis.py
import re
import math
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon', quiet=True)

FILLER_WORDS = set([
    "um", "uh", "like", "you know", "i mean", "so", "actually",
    "basically", "right", "well", "hmm", "huh", "erm"
])

sia = SentimentIntensityAnalyzer()

# -------------------------
# Existing utility functions
# -------------------------
def count_words(text):
    tokens = re.findall(r"\w+", text)
    return len(tokens), tokens

def words_per_minute(total_words, duration_seconds):
    if not duration_seconds or duration_seconds <= 0:
        return 0.0
    minutes = duration_seconds / 60.0
    return total_words / minutes

def filler_stats(text):
    lower = text.lower()
    counts = {}
    total = 0
    for f in FILLER_WORDS:
        c = lower.count(f)
        if c > 0:
            counts[f] = c
            total += c
    return {"total": total, "by_word": counts}

def sentiment_summary(text):
    scores = sia.polarity_scores(text)
    return scores

def per_segment_sentiment(segments):
    out = []
    for s in segments:
        scores = sia.polarity_scores(s["text"])
        out.append({
            "start": s["start"],
            "end": s["end"],
            "text": s["text"],
            "sentiment": scores
        })
    return out

def keyword_coverage(user_keywords, transcript_text):
    lower = transcript_text.lower()
    found = []
    missing = []
    for k in user_keywords:
        if k.strip().lower() in lower:
            found.append(k)
        else:
            missing.append(k)
    return {"found": found, "missing": missing}

def compute_overall_score(wpm, filler_total, sentiment_compound):
    score = 50.0

    # WPM contribution (scale 0-20)
    if wpm <= 0:
        wpm_score = 0
    else:
        if 120 <= wpm <= 160:
            wpm_score = 20
        else:
            d = min(abs(wpm - 140), 100)
            wpm_score = max(0, 20 - (d / 5.0))
    score += wpm_score

    # filler penalty (scale -20 to 0)
    filler_penalty = min(20, filler_total * 2)
    score -= filler_penalty

    # sentiment bonus (scale -10 to +10)
    sentiment_bonus = max(-10, min(10, sentiment_compound * 10))
    score += sentiment_bonus

    score = max(0, min(100, score))
    return round(score, 1)

# -------------------------
# NEW functions (add-ons)
# -------------------------
def generate_summary(transcript, wpm, sentiment, filler_total, round_type):
    """
    Produce a concise summary describing pace, filler use, and sentiment,
    tuned by interview round_type.
    """
    mood = "positive" if sentiment["compound"] > 0.2 else \
           "neutral" if sentiment["compound"] > -0.2 else "negative"

    pace_desc = (
        "fast-paced" if wpm > 160 else
        "slow-paced" if wpm < 110 else
        "well-paced"
    )

    filler_desc = (
        "high filler usage" if filler_total > 20 else
        "moderate filler usage" if filler_total > 5 else
        "minimal filler usage"
    )

    role_hint = ""
    if round_type and round_type.lower() in ["technical", "tech"]:
        role_hint = " Responses could benefit from more concrete technical examples and clarity on design choices."
    elif round_type and round_type.lower() in ["hr", "behavioral"]:
        role_hint = " Consider focusing on structured storytelling (STAR) for behavioral answers."
    elif round_type and round_type.lower() in ["managerial"]:
        role_hint = " Consider emphasizing leadership decisions, outcomes, and stakeholder impact."

    summary = (
        f"This appears to be a {round_type} interview. The candidate spoke in a {pace_desc} manner "
        f"with {filler_desc}. Overall sentiment was {mood}. {role_hint}"
    )
    return summary

def improvement_suggestions(wpm, filler_total, sentiment):
    suggestions = []

    if wpm > 160:
        suggestions.append("Reduce speaking speed to improve clarity and allow listeners to follow.")
    elif wpm < 110:
        suggestions.append("Try to speak slightly faster and add more energy to your delivery.")

    if filler_total > 15:
        suggestions.append("Work on removing filler words such as 'um', 'uh', and 'like' — practice pauses instead.")
    elif filler_total > 5:
        suggestions.append("Be mindful of filler words; aim to reduce them with focused practice.")

    if sentiment["compound"] < -0.2:
        suggestions.append("Aim for a more positive and confident tone when responding.")
    elif sentiment["compound"] > 0.5:
        suggestions.append("Your tone is positive — maintain this while balancing clarity.")

    if not suggestions:
        suggestions.append("Good communication. Small refinements (pauses, clarity) may help make it stronger.")

    return suggestions

def confidence_score(filler_total, sentiment, wpm):
    """
    Derive a confidence score (0-100) from simple heuristics:
    - fewer fillers -> higher
    - positive sentiment -> higher
    - reasonable WPM -> slightly higher
    """
    score = 60

    score -= min(20, filler_total)  # filler penalty

    # sentiment contributes: compound in [-1,1] → scale +/-20
    score += int(sentiment.get("compound", 0.0) * 20)

    # pace influence
    if wpm < 110 or wpm > 180:
        score -= 8
    else:
        score += 5

    score = max(0, min(100, int(round(score))))
    return score

def tone_label(sentiment, filler_total):
    """
    Simple rule-based tone label:
    - Confident: positive sentiment and low fillers
    - Nervous: negative sentiment and moderate/high fillers
    - Uncertain: many fillers
    - Neutral: otherwise
    """
    c = sentiment.get("compound", 0.0)
    if c > 0.4 and filler_total < 10:
        return "Confident"
    if c < -0.2 and filler_total > 10:
        return "Nervous"
    if filler_total > 25:
        return "Uncertain"
    return "Neutral"

def extract_key_points(transcript, max_points=5):
    """
    Lightweight key-point extraction:
    - split into sentences and pick longer sentences as likely key statements
    - returns top `max_points` candidate sentences
    """
    if not transcript or not transcript.strip():
        return []

    # naive sentence split
    sentences = re.split(r'(?<=[.!?])\s+', transcript.strip())
    # score sentences by length and presence of keywords (numbers, 'project', 'lead', 'designed', etc.)
    keywords = ["project", "designed", "implemented", "led", "improved", "reduced", "increased", "result", "achieve"]
    scored = []
    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        score = len(s_clean.split())
        for k in keywords:
            if k in s_clean.lower():
                score += 5
        scored.append((score, s_clean))

    scored.sort(reverse=True, key=lambda x: x[0])
    top = [s for _, s in scored[:max_points]]
    return top
