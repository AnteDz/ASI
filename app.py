import streamlit as st
import pandas as pd
import numpy as np
import joblib
from autogluon.tabular import TabularPredictor

st.set_page_config(page_title="Car Price Predictor", layout="wide")

raw = pd.read_csv("data/01_raw/Car_Prices_Poland_Kaggle.csv", encoding="utf-8")
raw.drop(
    columns=[c for c in raw.columns if c.lower().startswith("unnamed")] + ["province"],
    errors="ignore",
    inplace=True
)

all_marks = sorted(raw["mark"].dropna().unique())
model_map = raw.groupby("mark")["model"].apply(lambda s: sorted(s.dropna().unique())).to_dict()
gen_map = raw.groupby("model")["generation_name"] \
    .apply(lambda s: sorted(s.dropna().str.replace(r"^gen-", "", regex=True).unique())) \
    .to_dict()
all_cities = sorted(raw["city"].dropna().unique())


@st.cache_resource
def load_preprocessors(path="data/06_models/preprocessors.pkl"):
    return joblib.load(path)


preproc = load_preprocessors()
scaler = preproc["scaler"]
gen_le = preproc["gen_le"]
model_te_map = preproc["model_te_map"]
top_marks = preproc["top_marks"]
top_cities = preproc["top_cities"]


@st.cache_resource
def load_predictor(path="data/07_model_output/car_price_predictor_final"):
    return TabularPredictor.load(path)


predictor = load_predictor()

template_cols = list(pd.read_csv("data/02_intermediate/features.csv", nrows=0).columns)

st.title("🛻 Car Price Predictor")

st.subheader("Wprowadź dane samochodu:")
c1, c2, c3 = st.columns(3)
with c1:
    mark = st.selectbox("Marka", all_marks)
    model = st.selectbox("Model", model_map.get(mark, ["unknown"]))
    year = st.number_input("Rok produkcji", 1990, 2025, 2015)
with c2:
    mileage = st.number_input("Przebieg [km]", 2000, 300000, 50000)
    vol = st.number_input("Pojemność silnika", 400, 6000, 2000)
    fuel = st.selectbox("Paliwo", ["Gasoline", "Diesel"])
with c3:
    gen = st.selectbox("Generacja", gen_map.get(model, ["unknown"]))
    city = st.selectbox("Miasto", all_cities)

if st.button("Oblicz cenę"):
    df = pd.DataFrame([{
        "mark": mark,
        "model": model,
        "year": year,
        "mileage": mileage,
        "vol_engine": vol,
        "fuel": fuel,
        "generation_name": gen,
        "city": city
    }])

    df["fuel_encoded"] = df["fuel"].map({"Gasoline": 0, "Diesel": 1})
    df["generation_name"] = df["generation_name"].fillna("unknown")
    df["age"] = 2025 - df["year"]
    df["mileage_per_year"] = df["mileage"] / df["age"].replace(0, np.nan)
    df["log_mileage"] = np.log1p(df["mileage"])

    num_cols = ["age", "mileage", "mileage_per_year", "vol_engine", "log_mileage"]
    df[num_cols] = scaler.transform(df[num_cols])

    df["model_te"] = df["model"].map(model_te_map) \
        .fillna(np.mean(list(model_te_map.values())))

    df["mark_group"] = df["mark"].where(df["mark"].isin(top_marks), "other_mark")
    df = pd.get_dummies(df, columns=["mark_group"], prefix="mark")

    df["city_group"] = df["city"].where(df["city"].isin(top_cities), "other_city")
    df = pd.get_dummies(df, columns=["city_group"], prefix="city")

    raw_gen = df["generation_name"].str.replace(r"^gen-", "", regex=True).fillna("unknown")
    gen_grouped = raw_gen.where(raw_gen.isin(gen_le.classes_), "other")
    df["generation_name_encoded"] = gen_le.transform(gen_grouped)
    df = pd.get_dummies(df, columns=["generation_name_encoded"], prefix="gen")

    drop = ["mark", "model", "year", "mileage", "fuel", "city", "generation_name", "fuel_encoded"]
    X_inf = df.drop(columns=drop, errors="ignore") \
        .reindex(columns=template_cols, fill_value=0)

    price = predictor.predict(X_inf).iloc[0]
    st.success(f"Przewidywana cena: {price:,.0f} PLN")
