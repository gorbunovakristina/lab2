"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput

import pandas as pd


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt.

    Разделите наблюдения по полу и для каждой группы вычислите корреляции
    признаков FSIQ, VIQ, PIQ, Weight, Height с MRI_Count методом Пирсона.
    В strongest_mri_feature верните название признака с наибольшим модулем
    корреляции с MRI_Count среди объединённых результатов двух групп.
    """
    df = pd.read_csv(data.csv_path, sep='\t', na_values=['NA'])
    men_count = int((df['Gender'] == 'Male').sum())
    women_count = int((df['Gender'] == 'Female').sum())

    features = ['FSIQ', 'VIQ', 'PIQ', 'Weight', 'Height']

    women_df = df[df['Gender'] == 'Female']
    women_mri_correlation = women_df[features].corrwith(women_df['MRI_Count']).to_dict()

    men_df = df[df['Gender'] == 'Male']
    men_mri_correlation = men_df[features].corrwith(men_df['MRI_Count']).to_dict()

    women_men_correlation = (list(women_mri_correlation.items()) + list(men_mri_correlation.items()))
    valid_correlation = [item for item in women_men_correlation if pd.notna(item[1])]
    max_correlation = max(valid_correlation, key=lambda item: abs(item[1]))
    strongest_mri_feature = max_correlation[0]

    return BrainCorrelationSummary(men_count=men_count,
                                   women_count=women_count,
                                   women_mri_correlation=women_mri_correlation,
                                   men_mri_correlation=men_mri_correlation,
                                   strongest_mri_feature=strongest_mri_feature)
