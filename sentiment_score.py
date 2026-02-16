import re
import pandas as pd
from transformers import pipeline

# ---- CONFIG ----
INPUT_CSV  = "hespress_culture_body.csv"
OUTPUT_CSV = "hespress_sentiment_transformer.csv"

MODEL_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
# If this model causes issues, tell me and we switch.

# ---- LOAD ----
df = pd.read_csv(INPUT_CSV)

clf = pipeline(
    "text-classification",
    model=MODEL_NAME,
    return_all_scores=True
)

def normalize_ws(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()

def chunk_text(text: str, max_chars: int = 900):
    text = normalize_ws(text)
    if not text:
        return []
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

def scores_to_scalar(model_output):
    """
    Robustly convert model output into a single scalar.
    Works whether the pipeline returns:
      - [{'label': 'POS', 'score': 0.8}, {'label': 'NEG', 'score': 0.2}]   (list of dicts)
      - {'label': 'POS', 'score': 0.8}                                    (single dict)
      - or nested variants.
    Returns: P(pos) - P(neg). Neutral (if present) contributes 0.
    """

    # Case 1: output is a dict like {'label': 'POS', 'score': 0.8}
    if isinstance(model_output, dict):
        label = str(model_output.get("label", "")).lower()
        score = float(model_output.get("score", 0.0))
        if "pos" in label or "positive" in label:
            return score
        if "neg" in label or "negative" in label:
            return -score
        return 0.0  # neutral/other

    # Case 2: output is a list
    if isinstance(model_output, list):
        # Sometimes it's a list of dicts (good), sometimes nested
        if len(model_output) == 0:
            return 0.0

        # If it's nested like [[{...},{...}]], flatten one level
        if isinstance(model_output[0], list):
            model_output = model_output[0]

        # If it's a list of dicts, build pos/neg probs
        if all(isinstance(x, dict) for x in model_output):
            label_map = {str(d.get("label","")).lower(): float(d.get("score",0.0)) for d in model_output}
            pos = 0.0
            neg = 0.0
            for k, v in label_map.items():
                if "pos" in k or "positive" in k:
                    pos = v
                elif "neg" in k or "negative" in k:
                    neg = v
            return pos - neg

    # Fallback: unknown format
    return 0.0

def score_article(text: str) -> float:
    chunks = chunk_text(text)
    if not chunks:
        return 0.0

    vals = []
    for ch in chunks:
        out = clf(ch)
        vals.append(scores_to_scalar(out))

    return sum(vals) / len(vals)


# ---- SCORE ----
df["sentiment_score"] = df["Body"].apply(score_article)
df["sentiment_intensity"] = df["sentiment_score"].abs()

# ---- SAVE ----
out = df[["doc_id", "sentiment_score", "sentiment_intensity"]]
out.to_csv(OUTPUT_CSV, index=False)

print(out.head())
print(f"Saved: {OUTPUT_CSV}")
