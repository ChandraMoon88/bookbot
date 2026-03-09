"""
services/review_service/sentiment.py
--------------------------------------
Analyses the sentiment of a review text.
Uses a simple VADER-based approach (no GPU required) with a
transformers fallback if the BERT model is available.
"""

import logging
from typing import Optional

log = logging.getLogger(__name__)


def analyse(text: str) -> dict:
    """
    Returns:
        sentiment   str  "positive" | "neutral" | "negative"
        score       float  -1.0 to 1.0
        stars       int   1-5
    """
    # Try transformers sentiment pipeline first
    try:
        from transformers import pipeline as hf_pipeline
        _clf = hf_pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
        )
        result  = _clf(text[:512])[0]
        label   = result["label"].lower()   # "positive" | "negative"
        conf    = result["score"]
        score   = conf if label == "positive" else -conf
        stars   = _score_to_stars(score)
        sentiment = "positive" if score > 0.1 else ("negative" if score < -0.1 else "neutral")
        return {"sentiment": sentiment, "score": round(score, 3), "stars": stars}
    except Exception as exc:
        log.debug("Transformers sentiment unavailable: %s – falling back to VADER", exc)

    # VADER fallback
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyser = SentimentIntensityAnalyzer()
        vs       = analyser.polarity_scores(text)
        score    = vs["compound"]
        if score >= 0.05:
            sentiment = "positive"
        elif score <= -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        return {"sentiment": sentiment, "score": round(score, 3), "stars": _score_to_stars(score)}
    except ImportError:
        pass

    # Bare keyword fallback
    positive_words = {"excellent", "great", "amazing", "love", "perfect", "wonderful", "fantastic"}
    negative_words = {"terrible", "awful", "horrible", "worst", "bad", "dirty", "rude", "disappointing"}
    tokens = set(text.lower().split())
    pos = len(tokens & positive_words)
    neg = len(tokens & negative_words)
    if pos > neg:
        return {"sentiment": "positive", "score": 0.5, "stars": 4}
    if neg > pos:
        return {"sentiment": "negative", "score": -0.5, "stars": 2}
    return {"sentiment": "neutral", "score": 0.0, "stars": 3}


def _score_to_stars(score: float) -> int:
    if score >= 0.6:
        return 5
    if score >= 0.2:
        return 4
    if score >= -0.2:
        return 3
    if score >= -0.6:
        return 2
    return 1
