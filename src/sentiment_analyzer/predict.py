import argparse
from pathlib import Path

import joblib


def predict(model_path: Path, text: str) -> tuple[str, float]:
    pipeline = joblib.load(model_path)
    label = pipeline.predict([text])[0]

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba([text])[0]
        confidence = float(probabilities.max())
    else:
        confidence = 0.0

    return label, confidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict sentiment for one review.")
    parser.add_argument("--model-path", default="models/sentiment_model.joblib", help="Path to trained model.")
    parser.add_argument("--text", required=True, help="Review text.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label, confidence = predict(Path(args.model_path), args.text)
    print(f"Текст: {args.text}")
    print(f"Тональность: {label}")
    print(f"Уверенность: {confidence:.2f}")


if __name__ == "__main__":
    main()
