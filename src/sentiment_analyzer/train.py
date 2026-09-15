import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sentiment_analyzer.preprocess import clean_text


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=clean_text,
                    ngram_range=(1, 2),
                    min_df=1,
                ),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def train(data_path: Path, model_path: Path, report_path: Path) -> dict:
    df = pd.read_csv(data_path)

    required_columns = {"text", "sentiment"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing_columns))}")

    df = df.dropna(subset=["text", "sentiment"]).copy()
    if df.empty:
        raise ValueError("Dataset is empty after dropping missing values.")

    x_train, x_test, y_train, y_test = train_test_split(
        df["text"],
        df["sentiment"],
        test_size=0.25,
        random_state=42,
        stratify=df["sentiment"],
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "train_size": len(x_train),
        "test_size": len(x_test),
        "classification_report": report,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_path)
    report_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train review sentiment classifier.")
    parser.add_argument("--data", default="data/sample_reviews.csv", help="Path to CSV with text and sentiment columns.")
    parser.add_argument("--model-path", default="models/sentiment_model.joblib", help="Where to save trained model.")
    parser.add_argument("--report-path", default="reports/metrics.json", help="Where to save metrics report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train(Path(args.data), Path(args.model_path), Path(args.report_path))
    print(f"Model trained. Accuracy: {metrics['accuracy']:.3f}")
    print(f"Saved model to: {args.model_path}")
    print(f"Saved report to: {args.report_path}")


if __name__ == "__main__":
    main()
