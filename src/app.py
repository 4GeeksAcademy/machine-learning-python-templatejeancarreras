"""
src/app.py
──────────
Pipeline de producción para análisis de sentimiento en reseñas de clientes.

Usa el modelo nlptown/bert-base-multilingual-uncased-sentiment (Hugging Face)
para clasificar cada reseña en 1–5 estrellas y en bandas de sentimiento.

Entrada:  data/raw/reviews.csv
Salida:   data/processed/reviews_with_sentiment.csv
"""

import os
import sys
import warnings
from pathlib import Path

import pandas as pd
from transformers import pipeline

warnings.filterwarnings("ignore")

# ─── Rutas ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "reviews.csv"
OUT_PATH = ROOT / "data" / "processed" / "reviews_with_sentiment.csv"

# ─── Modelo (pinned — no depende de "latest") ─────────────────────
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"


def load_data() -> pd.DataFrame:
    """Lee el CSV de reseñas."""
    if not RAW_PATH.exists():
        print(f"[ERROR] No se encuentra {RAW_PATH}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(RAW_PATH)
    print(f"[OK] {len(df)} reseñas cargadas desde {RAW_PATH}")
    return df


def load_model():
    """Carga el pipeline de sentiment-analysis una sola vez (fuera del loop)."""
    print(f"[...] Cargando modelo {MODEL_NAME} …")
    pipe = pipeline("sentiment-analysis", model=MODEL_NAME, tokenizer=MODEL_NAME)
    print("[OK] Modelo listo")
    return pipe


def predict_sentiment(pipe, text: str):
    """
    Clasifica un texto y devuelve:
        label_raw, score, stars, sentiment_band
    """
    result = pipe(text[:512])[0]                     # truncar a 512 tokens
    label = result["label"]
    score = result["score"]
    stars = int(label.split()[0])                    # "3 stars" → 3

    if stars <= 2:
        band = "Negativo"
    elif stars == 3:
        band = "Neutral"
    else:
        band = "Positivo"

    return label, round(score, 4), stars, band


def main():
    # 1. Cargar datos
    df = load_data()

    # 2. Cargar modelo (una sola vez)
    pipe = load_model()

    # 3. Inferencia sobre todas las reseñas
    print(f"[...] Procesando {len(df)} reseñas …")
    results = [predict_sentiment(pipe, text) for text in df["review_text"]]

    df[["pred_label", "pred_score", "pred_stars", "sentiment_band"]] = (
        pd.DataFrame(results, index=df.index)
    )

    # 4. Mostrar resumen
    total = len(df)
    pos = (df["pred_stars"] >= 4).sum()
    neu = (df["pred_stars"] == 3).sum()
    neg = (df["pred_stars"] <= 2).sum()
    print(f"\n{'═' * 55}")
    print("  DISTRIBUCIÓN DE SENTIMIENTO (modelo)")
    print(f"{'═' * 55}")
    print(f"  Positivo (4–5⭐): {pos:>3} ({pos / total * 100:.1f}%)")
    print(f"  Neutral  (3⭐) : {neu:>3} ({neu / total * 100:.1f}%)")
    print(f"  Negativo (1–2⭐): {neg:>3} ({neg / total * 100:.1f}%)")
    print(f"  Promedio predicho: {df['pred_stars'].mean():.2f} ⭐")
    print(f"{'═' * 55}\n")

    # 5. Guardar resultados
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"[OK] Resultados guardados en {OUT_PATH}")


if __name__ == "__main__":
    main()
