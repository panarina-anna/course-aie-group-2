from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2

def test_compute_quality_flags_constant_columns():
    """Тест для эвристики has_constant_columns"""
    df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'constant_col': [42, 42, 42, 42, 42], 
        'variable_col': [1, 2, 3, 4, 5],
        'category': ['A', 'B', 'C', 'D', 'E']
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)
    
    assert flags['has_constant_columns'] == True
    assert 'constant_columns' in flags
    assert 'constant_col' in flags['constant_columns']
    assert flags['n_constant_columns'] == 1
    
    assert flags['quality_score'] < 1.0


def test_compute_quality_flags_high_cardinality_categoricals():
    """Тест для эвристики has_high_cardinality_categoricals"""
    df = pd.DataFrame({
        'id': range(10),
        'high_card_col': [f'category_{i}' for i in range(10)],
        'low_card_col': ['A', 'B'] * 5,
        'numeric': range(10)
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)
    
    assert flags['has_high_cardinality_categoricals'] == True
    assert 'high_cardinality_categoricals' in flags
    assert len(flags['high_cardinality_categoricals']) > 0
    
    problem_cols = [item['column_name'] for item in flags['high_cardinality_categoricals']]
    assert 'high_card_col' in problem_cols
    assert 'low_card_col' not in problem_cols 


def test_compute_quality_flags_suspicious_id_duplicates():
    """Тест для эвристики has_suspicious_id_duplicates"""
    df = pd.DataFrame({
        'user_id': [1, 2, 3, 3, 4],
        'customer_id': [101, 102, 103, 104, 105],
        'item_id': [1001, 1002, 1002, 1003, 1004],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve']
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)
    
    assert flags['has_suspicious_id_duplicates'] == True
    assert 'suspicious_id_duplicates' in flags
    assert len(flags['suspicious_id_duplicates']) > 0
    
    problem_cols = [item['column_name'] for item in flags['suspicious_id_duplicates']]
    assert 'user_id' in problem_cols or 'item_id' in problem_cols


def test_compute_quality_flags_many_zero_values():
    """Тест для эвристики has_many_zero_values"""
    df = pd.DataFrame({
        'id': range(10),
        'many_zeros': [0, 0, 0, 0, 0, 1, 2, 3, 4, 5],
        'few_zeros': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'no_zeros': range(1, 11),
        'string_col': ['A', 'B', 'C'] * 3 + ['D']
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)
    
    assert flags['has_many_zero_values'] == True
    assert 'many_zero_values' in flags
    assert len(flags['many_zero_values']) > 0
    
    problem_cols = [item['column_name'] for item in flags['many_zero_values']]
    assert 'many_zeros' in problem_cols
    assert 'few_zeros' not in problem_cols  
    assert 'no_zeros' not in problem_cols 


def test_compute_quality_flags_cli_parameters():
    """Тест для проверки передачи параметров из CLI"""
    df = pd.DataFrame({
        'col1': [1, 2, 3, 4, 5],
        'col2': [0, 0, 0, 1, 2],
        'category': ['A', 'B', 'C', 'D', 'E']
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    
    flags = compute_quality_flags(
        summary, 
        missing_df, 
        df,
        min_missing_threshold=0.2,
        high_cardinality_pct=0.5,
        zero_threshold=0.5
    )
    
    assert flags['cli_min_missing_threshold'] == 0.2
    assert flags['cli_high_cardinality_pct'] == 0.5
    assert flags['cli_zero_threshold'] == 0.5
    
    assert flags['has_many_zero_values'] == True
    
    assert flags['has_high_cardinality_categoricals'] == True


def test_compute_quality_flags_all_new_heuristics():
    """Комплексный тест для всех новых эвристик"""
    df = pd.DataFrame({
        'id': [1, 2, 3, 3, 4],
        'constant': [5, 5, 5, 5, 5],
        'high_card': [f'cat_{i}' for i in range(5)],
        'many_zeros': [0, 0, 0, 1, 2],
        'normal': [1.1, 2.2, 3.3, 4.4, 5.5]
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)
    
    assert flags['has_constant_columns'] == True
    assert flags['has_high_cardinality_categoricals'] == True
    assert flags['has_suspicious_id_duplicates'] == True
    assert flags['has_many_zero_values'] == True
    
    assert len(flags['constant_columns']) > 0
    assert len(flags['high_cardinality_categoricals']) > 0
    assert len(flags['suspicious_id_duplicates']) > 0
    assert len(flags['many_zero_values']) > 0
    
    assert 0.0 <= flags['quality_score'] <= 1.0
    assert flags['quality_score'] < 1.0