# sentiment.py
import ollama
from typing import List, Dict

class SentimentAnalyzer:
    def __init__(self, model_name: str = "gemma3:270m"):
        self.model_name = model_name

    def analyze(self, comments: List[str]) -> Dict:
        results = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0

        # Helper to normalize labels
        def normalize_label(raw: str) -> str:
            raw = raw.lower()
            if "positive" in raw:
                return "Positive"
            elif "negative" in raw:
                return "Negative"
            else:
                return "Neutral"

        for comment in comments:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a sentiment analysis assistant. "
                                   "Classify the following text strictly as Positive, Negative, or Neutral."
                    },
                    {"role": "user", "content": comment}
                ]
            )

            raw_sentiment = response['message']['content'].strip()
            sentiment_label = normalize_label(raw_sentiment)

            if sentiment_label == "Positive":
                positive_count += 1
            elif sentiment_label == "Negative":
                negative_count += 1
            else:
                neutral_count += 1

            results.append({
                "comment": comment,
                "label": sentiment_label
            })

        total = len(comments)
        summary = {
            "positive": positive_count,
            "negative": negative_count,
            "neutral": neutral_count
        } if total > 0 else {"positive": 0, "negative": 0, "neutral": 0}

        return {"summary": summary, "details": results}
