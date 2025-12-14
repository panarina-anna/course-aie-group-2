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
