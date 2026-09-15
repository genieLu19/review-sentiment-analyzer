# Review Sentiment Analyzer

**Review Sentiment Analyzer** - ML-проект для анализа тональности пользовательских отзывов. Модель определяет, является ли отзыв положительным, нейтральным или негативным.

Проект сделан как готовый GitHub-кейс для портфолио Data Scientist / ML Engineer: здесь есть данные, обучение модели, сохранение артефакта, предсказания из командной строки, метрики качества и тесты.

## Что умеет проект

- очищает текст отзывов;
- обучает модель классификации тональности;
- сравнивает реальные и предсказанные классы;
- сохраняет обученную модель в файл;
- делает предсказание для нового отзыва;
- сохраняет отчет с метриками в `reports/metrics.json`.

## Задача машинного обучения

Тип задачи: **многоклассовая классификация текста**.

Классы:

- `positive` - положительный отзыв;
- `neutral` - нейтральный отзыв;
- `negative` - негативный отзыв.

Модель получает текст отзыва и возвращает класс тональности.

## Структура проекта

```text
review-sentiment-analyzer/
├── data/
│   └── sample_reviews.csv
├── models/
├── reports/
├── src/
│   └── sentiment_analyzer/
│       ├── __init__.py
│       ├── predict.py
│       ├── preprocess.py
│       └── train.py
├── tests/
│   └── test_preprocess.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Как запустить

Открой PowerShell в папке проекта:

```powershell
cd "путь\к\review-sentiment-analyzer"
```

Создай виртуальное окружение:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Обучи модель:

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m sentiment_analyzer.train --data data/sample_reviews.csv --model-path models/sentiment_model.joblib --report-path reports/metrics.json
```

Сделай предсказание:

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m sentiment_analyzer.predict --model-path models/sentiment_model.joblib --text "Книга понравилась, читается легко и очень увлекает"
```

Запусти тесты:

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m pytest
```

## Пример результата

```text
Текст: Книга понравилась, читается легко и очень увлекает
Тональность: positive
Уверенность: 0.91
```

## Метрики

После обучения проект сохраняет отчет:

```text
reports/metrics.json
```

В отчете есть:

- `accuracy`;
- `precision`;
- `recall`;
- `f1-score`;
- количество объектов в train/test выборках.

## Технологии

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Logistic Regression
- Joblib
- Pytest

## Что показывает этот кейс

Этот проект демонстрирует навыки, которые важны для Data Scientist / ML Engineer:

- подготовка текстовых данных;
- построение ML-пайплайна;
- обучение модели классификации;
- оценка качества через метрики;
- сохранение и повторное использование модели;
- оформление проекта для GitHub и резюме.

## Идеи для развития

- добавить больший датасет отзывов;
- сделать Streamlit-интерфейс;
- добавить сравнение нескольких моделей;
- добавить обработку английских и русских отзывов;
- построить confusion matrix;
- добавить Dockerfile для запуска в контейнере.
