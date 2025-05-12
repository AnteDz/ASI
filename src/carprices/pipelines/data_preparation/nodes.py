import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(path: str) -> pd.DataFrame:
    """
    Wczytuje dane z pliku CSV i usuwa niepotrzebne kolumny.

    Parametry:
    ----------
    path: str
        Ścieżka do pliku CSV.

    Zwraca:
    -------
    pd.DataFrame
        Załadowany DataFrame.
    """
    df = pd.read_csv(path)
    # Usuń kolumnę indeksu, jeśli istnieje
    df = df.drop(columns=[col for col in df.columns if col.lower().startswith('unnamed')], errors='ignore')
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Czyści dane: usuwa duplikaty, wypełnia brakujące wartości i filtruje nieprawidłowe rekordy.

    Parametry:
    ----------
    df: pd.DataFrame
        DataFrame do oczyszczenia.

    Zwraca:
    -------
    pd.DataFrame
        Oczyszczony DataFrame.
    """
    # Usuń duplikaty
    df = df.drop_duplicates().copy()

    # Wypełnij brakujące generation_name
    if 'generation_name' in df.columns:
        df['generation_name'] = df['generation_name'].fillna('unknown')

    # Filtruj nieprawidłowe wartości
    df = df[(df['year'] > 1900) &
            (df['mileage'] >= 0) &
            (df['vol_engine'] > 0) &
            (df['price'] > 0)]

    return df


def feature_engineering(df: pd.DataFrame, current_year: int = 2025) -> pd.DataFrame:
    """
    Tworzy dodatkowe cechy: wiek pojazdu, logarytm przebiegu oraz koduje kategorie.

    Parametry:
    ----------
    df: pd.DataFrame
        Oczyszczony DataFrame.
    current_year: int, opcjonalnie
        Rok referencyjny do obliczania wieku (domyślnie 2025).

    Zwraca:
    -------
    pd.DataFrame
        DataFrame z nowymi cechami i zakodowanymi zmiennymi.
    """
    df = df.copy()

    # Wiek samochodu
    df['age'] = current_year - df['year']

    # Logarytm przebiegu (unikanie log(0))
    df['log_mileage'] = np.log1p(df['mileage'])

    # Lista cech kategorycznych do zakodowania
    cat_cols = [col for col in ['mark', 'model', 'generation_name', 'fuel', 'province'] if col in df.columns]

    # One-hot encoding
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Usuń kolumny oryginalne niewymagane
    df = df.drop(columns=['year', 'mileage'], errors='ignore')

    return df


def prepare_for_autogluon(path: str, test_size: float = 0.2, random_state: int = 42) -> pd.DataFrame:
    """
    Kompletna funkcja od wczytania przez oczyszczenie i inżynierię cech,
    gotowa do podania do AutoGluon.TabularPredictor.fit.

    Parametry:
    ----------
    path: str
        Ścieżka do pliku CSV.
    test_size: float, opcjonalnie
        Procent danych testowych (niekonieczne, AutoGluon może sam podzielić).
    random_state: int, opcjonalnie
        Ziarno losowe przy dzieleniu (jeśli użyjesz ręcznego splitu).

    Zwraca:
    -------
    pd.DataFrame
        Gotowy DataFrame z cechami oraz kolumną 'price'.
    """
    # Wczytaj i oczyść dane
    df = load_data(path)
    df = clean_data(df)

    # Stwórz cechy
    df_prepared = feature_engineering(df)

    # AutoGluon wymaga, by etykieta 'price' była w df:
    if 'price' not in df_prepared.columns:
        df_prepared['price'] = df['price'].values

    return df_prepared


def train_autogluon(df: pd.DataFrame, label: str = 'price', output_path: str = 'ag_models/'):
    """
    Przykładowa funkcja trenująca model AutoGluon.TabularPredictor.

    Parametry:
    ----------
    df: pd.DataFrame
        DataFrame z cechami i kolumną etykiety.
    label: str
        Nazwa kolumny etykiety.
    output_path: str, opcjonalnie
        Ścieżka, gdzie zapisać modele AutoGluon.

    Zwraca:
    -------
    predictor
        Wytrenowany obiekt TabularPredictor.
    """
    from autogluon.tabular import TabularPredictor

    predictor = TabularPredictor(label=label, path=output_path).fit(
        train_data=df,
        time_limit=600,         # czas treningu w sekundach
        presets='best_quality'
    )
    return predictor


def split_data(df: pd.DataFrame, label: str = 'price', test_size: float = 0.2, random_state: int = 42):
    """
    Opcjonalna funkcja do podziału danych na zbiór treningowy i testowy.

    Parametry:
    ----------
    df: pd.DataFrame
        DataFrame z etykietą.
    label: str
        Nazwa kolumny etykiety.
    test_size: float
        Ułamek danych testowych.
    random_state: int
        Ziarno losowe.

    Zwraca:
    -------
    X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[label])
    y = df[label]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
