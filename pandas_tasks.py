"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    """Выполните загрузку и анализ датасета Titanic.

    Нужно: посчитать пропуски, число пассажиров старше 30 лет, средний возраст
    и долю выживших по классам, а также пять наибольших тарифов по убыванию.
    """

    df = pd.read_csv(data.csv_path)

    # Считаем и сразу преобразовываем в словарь.
    missing_by_column = df.isna().sum().to_dict()

    # Так как будет возвращена последовательность из True/False,
    # то достаточно сложить
    adults_over_30_count = int((df['Age'] > 30).sum())

    # .mean() находит среднее
    mean_age_by_pclass = df.groupby('Pclass')['Age'].mean().to_dict()
    survival_rate_by_pclass = df.groupby('Pclass')['Survived'].mean().to_dict()

    # ascending отвечает за направление сортировки
    sorted_by_fare = df['Fare'].sort_values(ascending=False)
    highest_fares = sorted_by_fare.head(5).to_list()
    row_count = int(len(df))

    return TitanicSummary(row_count=row_count,
                          missing_by_column=missing_by_column,
                          adults_over_30_count=adults_over_30_count,
                          mean_age_by_pclass=mean_age_by_pclass,
                          survival_rate_by_pclass=survival_rate_by_pclass,
                          highest_fares=highest_fares)