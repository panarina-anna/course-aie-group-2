# S03 – eda_cli: мини-EDA для CSV

Небольшое CLI-приложение для базового анализа CSV-файлов.
Используется в рамках Семинара 03 курса «Инженерия ИИ».

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) установлен в систему

## Инициализация проекта

В корне проекта (S03):

```bash
uv sync
```

Эта команда:

- создаст виртуальное окружение `.venv`;
- установит зависимости из `pyproject.toml`;
- установит сам проект `eda-cli` в окружение.

## Запуск CLI

### Краткий обзор

```bash
uv run eda-cli overview data/example.csv
```

Параметры:

- `--sep` – разделитель (по умолчанию `,`);
- `--encoding` – кодировка (по умолчанию `utf-8`).

```bash
uv run eda-cli head data/my_data.csv --n 5
```

Параметры:

- `--sep` – разделитель (по умолчанию `,`);
- `--encoding` – кодировка (по умолчанию `utf-8`);
- `--n` - количество выводимых строк(по умолчанию `5`);
- `--show_header` - выводить заголовки или нет (по умолчанию `True`)

### Полный EDA-отчёт

```bash
uv run eda-cli report data/example.csv --out-dir reports
```

В результате в каталоге `reports/` появятся:

- `report.md` – основной отчёт в Markdown;
- `summary.csv` – таблица по колонкам;
- `missing.csv` – пропуски по колонкам;
- `correlation.csv` – корреляционная матрица (если есть числовые признаки);
- `top_categories/*.csv` – top-k категорий по строковым признакам;
- `hist_*.png` – гистограммы числовых колонок;
- `missing_matrix.png` – визуализация пропусков;
- `correlation_heatmap.png` – тепловая карта корреляций.

## Новые параметры команды report
- `max_hist_columns` - ограничивает количество создаваемых гистограмм для числовых колонок (по умолчанию `6`);

```bash
uv run eda-cli report data.csv --max-hist-columns 3
```

- `top_k_categories` - определяет количество топ-значений для категориальных признаков (по умолчанию `5`);

```bash
uv run eda-cli report data.csv --top-k-categories 10
```

- `title` - устанавливает заголовок в Markdown-отчёте;

```bash
uv run eda-cli report data.csv --title "Отчёт по продажам"
```

- `min_missing_share` - порог доли пропусков для пометки колонки как проблемной (по умолчанию `0.3`).

```bash
uv run eda-cli report data.csv --min-missing-share 0.2
```

## Тесты

```bash
uv run pytest -q
```

## HTTP-сервис
 
```bash
uv run uvicorn eda_cli.api:app --port 8000
```
Доступные эндпоинты:

- `GET /health` - проверка состояния сервиса

Возвращает: {"status": "ok", "service": "dataset-quality", "version": "0.2.0"}

- `POST /quality` - оценка по агрегированным признакам (n_rows, max_missing_share, …)

Принимает JSON: {"n_rows": 100, "n_cols": 10, "max_missing_share": 0.1, ...}

- `POST /quality-from-csv` - оценка качества по CSV-файлу

Принимает CSV-файл (multipart/form-data)

Использует EDA-ядро из HW03

Возвращает оценку и базовые флаги качества

- `POST /quality-flags-from-csv` - полный набор флагов качества (включая новые эвристики)

Принимает CSV-файл

Возвращает полный набор флагов с новыми эвристиками из HW03:

has_constant_columns - есть ли константные колонки

has_high_cardinality_categoricals - категории с высокой кардинальностью

has_suspicious_id_duplicates - подозрительные дубликаты ID

has_many_zero_values - много нулевых значений

- `GET /metrics` - статистика сервиса (запросы, latency, последний датасет)

Возвращает: количество запросов, среднее время выполнения, параметры последнего датасета