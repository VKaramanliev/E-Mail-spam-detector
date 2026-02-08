# E-mail Spam Detector

Този проект представлява **machine learning система за разпознаване на spam e-mail-и**.  
Той покрива целия процес от зареждане на данни, обучение на модел, оценка на качеството и реална употреба чрез команден ред и уеб интерфейс.

## Цел на проекта

Основната цел е да се изгради система, която:
- приема текст на e-mail
- определя дали е **spam** или **not spam (ham)**
- връща и вероятност за това решение

## Как работи моделът

1. **TF-IDF Vectorizer**
   - превръща текста в числови вектори
   - използва unigrams и bigrams

2. **Logistic Regression**
   - обучава се да класифицира spam / ham
   - връща и вероятности

Моделът е реализиран чрез `sklearn.pipeline.Pipeline`.

## Формат на данните

Очакваният CSV файл трябва да съдържа:

- колона `text` – текстът на e-mail-а
- колона `label` – `spam` или `ham`
- всеки ред представлява един e-mail

Проверките за коректност се извършват в `data.py`.

---

## Инсталация и стартиране

### 1. Клониране на проекта
```bash
git clone <repo-url>
cd email-spam-detector
```

### 2. Виртуална среда
```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows
```

### 3. Инсталиране на зависимостите
```bash
pip install -e
pip install -e ".[dev]"
```

## Обучение на модел (CLI):

Обучението се извършва чрез команден ред.

```bash
    spamdet-train --data data/raw/emails.csv --out models/spam_model.joblib
```

### Какво прави тази команда:
- зарежда CSV файла
- разделя данните на train / validation
- обучава ML модела
- оценява го
- запазва модела като `.joblib`

### Извеждани метрики:
- accuracy
- f1_macro
- classification report

## Класификация на текст (CLI)
    spamdet-predict --model models/spam_model.joblib --text "Congratulations! You won a prize!"


Резултат:
    label=spam spam_proba=0.9823


## Streamlit уеб приложение

### Стартиране:
    streamlit run src/streamlit_app.py

### Какво може да се прави през интерфейса:
    - зареждане / обучение на модел
    - качване на CSV
    - класифициране на текст
    - визуализация на резултати
    - генериране на графики
    - запис на отчети


## Оценка и визуализации

### При оценка се генерират:
    - текстов отчет (accuracy, f1_macro, report)
    - confusion matrix (PNG)
    - bar chart с precision по класове

### Файловете се записват в:
    artifacts/reports/

Графиките се генерират автоматично от plots.py.


## Тестване

### Стартиране на всички тестове:
    pytest


### Тестовете покриват:
    - зареждане на данни
    - ML pipeline
    - оценка
    - предсказване


## Използвани технологии:
- Python 3.10+
- scikit-learn
- pandas
- matplotlib
- joblib
- streamlit
- pytest

## Накратко
```bash
git clone <repo-url>
pip install -e ".[dev]"
spamdet-train --data data/raw/emails.csv --out models/spam_model.joblib
spamdet-predict --model models/spam_model.joblib --text "Congratulations! You won a prize!"
```