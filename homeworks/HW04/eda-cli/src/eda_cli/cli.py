from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from .core import (
    DatasetSummary,
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)
from .viz import (
    plot_correlation_heatmap,
    plot_missing_matrix,
    plot_histograms_per_column,
    save_top_categories_tables,
)

app = typer.Typer(help="Мини-CLI для EDA CSV-файлов")


def _load_csv(
    path: Path,
    sep: str = ",",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    if not path.exists():
        raise typer.BadParameter(f"Файл '{path}' не найден")
    try:
        return pd.read_csv(path, sep=sep, encoding=encoding)
    except Exception as exc:  # noqa: BLE001
        raise typer.BadParameter(f"Не удалось прочитать CSV: {exc}") from exc


@app.command()
def overview(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
) -> None:
    """
    Напечатать краткий обзор датасета:
    - размеры;
    - типы;
    - простая табличка по колонкам.
    """
    df = _load_csv(Path(path), sep=sep, encoding=encoding)
    summary: DatasetSummary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)

    typer.echo(f"Строк: {summary.n_rows}")
    typer.echo(f"Столбцов: {summary.n_cols}")
    typer.echo("\nКолонки:")
    typer.echo(summary_df.to_string(index=False))

# Вывод первых n строк
@app.command()
def head(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    n: int = typer.Option(5, help="Количество строк для вывода."),
    sep: str = typer.Option(",", help='Разделитель в CSV.'),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
    show_header: bool = typer.Option(True, help="Показывать заголовки колонок.")
    ) -> None:
        df = _load_csv(Path(path), sep=sep, encoding=encoding)

        if n < 0:
            typer.echo("Параметр n должен быть положительным.")
            raise typer.Exit(code=1)
        
        if n > len(df):
            typer.echo(f"Количество строк (n) должно быть меньше, чем число строк в файле (< {len(df)}).")
            n = len(df)

        typer.echo(f"Файл: {Path(path).name}")
        typer.echo(f"Количество строк: {len(df)}")
        typer.echo(f"Количество колонок: {len(df.columns)}")    
        typer.echo(f"Первые {n} строк:\n")   
        with pd.option_context('display.max_rows', n,
                                'display.max_columns', None,
                                'display.width', None,
                                'display.max_colwidth', 30):
            typer.echo(df.head(n).to_string(index=False, header=show_header)) 

@app.command()
def report(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    out_dir: str = typer.Option("reports", help="Каталог для отчёта."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
    max_hist_columns: int = typer.Option(6, help="Максимум числовых колонок для гистограмм."),
    top_k_categories: int = typer.Option(5, help="Количество топ-значений для категориальных признаков."),
    title: str = typer.Option("Анализ данных", help="Заголовок отчёта."),
    min_missing_share: float = typer.Option(0.3, help="Порог доли пропусков для пометки колонки как проблемной.")
) -> None:
    """
    Сгенерировать полный EDA-отчёт:
    - текстовый overview и summary по колонкам (CSV/Markdown);
    - статистика пропусков;
    - корреляционная матрица;
    - top-k категорий по категориальным признакам;
    - картинки: гистограммы, матрица пропусков, heatmap корреляции.
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df = _load_csv(Path(path), sep=sep, encoding=encoding)

    # 1. Обзор
    summary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)
    missing_df = missing_table(df)
    corr_df = correlation_matrix(df)
    top_cats = top_categories(df, top_k=top_k_categories)

    # 2. Качество в целом
    quality_flags = compute_quality_flags(
        summary,
        missing_df,
        df,
        min_missing_threshold=min_missing_share,
        high_cardinality_pct=0.3,
        zero_threshold=0.4
        )

    # 3. Сохраняем табличные артефакты
    summary_df.to_csv(out_root / "summary.csv", index=False)
    if not missing_df.empty:
        missing_df.to_csv(out_root / "missing.csv", index=True)
    if not corr_df.empty:
        corr_df.to_csv(out_root / "correlation.csv", index=True)
    save_top_categories_tables(top_cats, out_root / "top_categories", top_k=top_k_categories)

    # 4. Markdown-отчёт
    md_path = out_root / "report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"**Дата генерации:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Параметры анализа
        f.write("## Параметры анализа\n\n")
        f.write(f"- **Максимальное количество гистограмм:** `{max_hist_columns}`\n")
        f.write(f"- **Максимальное количество топ-категорий:** `{top_k_categories}`\n")
        f.write(f"- **Порог доли пропусков для проблемных колонок:** `{min_missing_share}`\n\n")

        f.write(f"**Исходный файл:** `{Path(path).name}`\n\n")
        f.write(f"**Строк:** {summary.n_rows}, **столбцов:** {summary.n_cols}\n\n")

        f.write("## Качество данных (эвристики)\n\n")
        f.write(f"- **Общая оценка качества:** {quality_flags['quality_score']:.2f}/1.0\n")
        f.write(f"- **Максимальная доля пропусков:** {quality_flags['max_missing_share']:.2%}\n")
        f.write(f"- **Слишком мало строк (<100):** {quality_flags['too_few_rows']}\n")
        f.write(f"- **Слишком много колонок (>100):** {quality_flags['too_many_columns']}\n")
        f.write(f"- **Слишком много пропусков (>50%):** {quality_flags['too_many_missing']}\n")
        
        # Новые эвристики
        f.write(f"- **Используемый порог пропусков:** {quality_flags['cli_min_missing_threshold']}\n")

        f.write(f"- **Колонок с пропусками > порога:** {quality_flags['n_problematic_by_threshold']}\n")
        if quality_flags['n_problematic_by_threshold'] > 0:
            f.write("  - Список:\n")
            for col, share in quality_flags['problematic_cols_by_threshold']:
                f.write(f"    - `{col}`: {share:.1%}\n")

        f.write(f"- **Количество константных колонок:** {quality_flags['n_constant_columns']}\n")

        f.write(f"- **Порог для нулевых значений:** {quality_flags['cli_zero_threshold']:.1%}\n")

        if 'zero_check_skipped' in quality_flags:
            f.write(f"- **Статус проверки нулей:** {quality_flags['zero_check_skipped']}\n")

        f.write(f"- **Есть константные колонки:** {quality_flags['has_constant_columns']}\n")
        if quality_flags['has_constant_columns']:
            const_cols = quality_flags.get('constant_columns', [])
            f.write(f"  - Константные колонки: {', '.join(const_cols)}\n")
        
        f.write(f"- **Есть категории с высокой кардинальностью:** {quality_flags['has_high_cardinality_categoricals']}\n")
        if quality_flags['has_high_cardinality_categoricals']:
            threshold = quality_flags.get('high_cardinality_threshold', 0)
            high_card_pct = quality_flags.get('cli_high_cardinality_pct', 0.3) * 100
            high_card_cols = quality_flags.get('high_cardinality_categoricals', [])
            f.write(f"  - Проблемные колонки (уникальных значений > {high_card_pct:.0f}):\n")
            for item in high_card_cols:
                percentage = (item['unique_values'] / summary.n_rows) * 100
                f.write(f"    - `{item['column_name']}`: {item['unique_values']} уникальных ({percentage:.0f}%)\n")
        
        f.write(f"- **Есть подозрительные дубликаты ID:** {quality_flags['has_suspicious_id_duplicates']}\n")
        if quality_flags['has_suspicious_id_duplicates']:
            f.write(f"  - Колонки с возможными дубликатами:\n")
            for item in quality_flags.get('suspicious_id_duplicates', []):
                f.write(f"    - `{item['column_name']}`: {item['unique_count']} уникальных ({item['unique_share']:.1%})\n")
        
        f.write(f"- **Есть колонки с большим количеством нулей:** {quality_flags['has_many_zero_values']}\n")
        if quality_flags['has_many_zero_values']:
            zero_thresh = quality_flags.get('cli_zero_threshold', 0.4)
            f.write(f"  - Колонки с > {zero_thresh:.0%} нулевых значений:\n")
            for item in quality_flags.get('many_zero_values', []):
                f.write(f"    - `{item['column_name']}`: {item['zero_share']:.1%} нулей\n")
        
        f.write("\n")

        f.write(f"## Колонки с большим количеством пропусков (порог > {min_missing_share})\n\n")
        problematic_cols = []
        for idx, row in missing_df.iterrows():
            if row['missing_share'] > min_missing_share:
                problematic_cols.append((idx, row['missing_share']))
        
        if problematic_cols:
            for col, share in problematic_cols:
                f.write(f"- `{col}`: {share:.1%} пропусков\n")
            f.write(f"\nВсего проблемных колонок: **{len(problematic_cols)}**\n\n")
        else:
            f.write("Нет колонок с превышением порога пропусков.\n\n")

        # Остальные разделы
        f.write("## Детальная информация по колонкам\n\n")
        f.write("Полная информация доступна в файле `summary.csv`.\n\n")

        f.write("## Анализ пропусков\n\n")
        if missing_df.empty:
            f.write("Пропуски не обнаружены или датасет пуст.\n\n")
        else:
            f.write("Детальная статистика пропусков в файле `missing.csv`.\n")
            f.write("Визуализация пропусков: `missing_matrix.png`\n\n")

        f.write("## Корреляционный анализ\n\n")
        if corr_df.empty:
            f.write("Недостаточно числовых колонок для анализа корреляции.\n\n")
        else:
            f.write("Корреляционная матрица в файле `correlation.csv`.\n")
            f.write("Тепловая карта корреляций: `correlation_heatmap.png`\n\n")

        f.write("## Анализ категориальных признаков\n\n")
        if not top_cats:
            f.write("Категориальные признаки не обнаружены.\n\n")
        else:
            f.write(f"Топ-{top_k_categories} значений для каждой категориальной колонки сохранены в папке `top_categories/`.\n\n")

        f.write("## Визуализации\n\n")
        num_numeric = len(df.select_dtypes(include='number').columns)
        actual_histograms = min(max_hist_columns, num_numeric)
        f.write(f"Гистограммы для {actual_histograms} из {num_numeric} числовых колонок.\n")
        f.write("См. файлы `hist_*.png`.\n")
    # 5. Картинки
    plot_histograms_per_column(df, out_root, max_columns=max_hist_columns)
    plot_missing_matrix(df, out_root / "missing_matrix.png")
    plot_correlation_heatmap(df, out_root / "correlation_heatmap.png")

    typer.echo(f"Отчёт сгенерирован в каталоге: {out_root}")
    typer.echo(f"- Основной markdown: {md_path}")
    typer.echo("- Табличные файлы: summary.csv, missing.csv, correlation.csv, top_categories/*.csv")
    typer.echo("- Графики: hist_*.png, missing_matrix.png, correlation_heatmap.png")


if __name__ == "__main__":
    app()
