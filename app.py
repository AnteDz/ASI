import streamlit as st
st.set_page_config(page_title="Car Price Predictor", layout="wide")

import pandas as pd
from autogluon.tabular import TabularPredictor

from src.carprices.pipelines.data_preparation.nodes import (
    create_numerical_features,
    scale_features,
)

def inference_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.drop(columns=['province'], errors='ignore', inplace=True)
    df['fuel_type'] = df['fuel']
    df = df[df['fuel_type'].isin(['Gasoline','Diesel'])].copy()
    df['fuel_encoded'] = df['fuel_type'].map({'Gasoline':0,'Diesel':1})
    df['generation_name'] = df['generation_name'].fillna('unknown')
    df['generation_name'] = df['generation_name'].str.replace(r'^gen-', '', regex=True)
    return df

def encode_inference(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    top_marks = [
        'audi','bmw','citroen','fiat','ford','honda','hyundai','kia','mazda',
        'mercedes-benz','mini','nissan','opel','other_mark','peugeot','renault',
        'seat','skoda','toyota','volkswagen','volvo'
    ]
    df['mark_group'] = df['mark'].where(df['mark'].isin(top_marks), 'other_mark')
    df = pd.get_dummies(df, columns=['mark_group'], prefix='mark')

    top_cities = [
        'Białystok','Bielany Wrocławskie','Bydgoszcz','Częstochowa','Elbląg',
        'Gdańsk','Gdynia','Gliwice','Gniezno','Katowice','Kielce','Kraków',
        'Kutno','Lublin','Mysłowice','Nowy Sącz','Olsztyn','Ostrów Mazowiecka',
        'Piaseczno','Poznań','Płock','Radom','Rybnik','Rzeszów','Szczecin',
        'Warszawa','Wrocław','Wągrowiec','Zabrze','Łódź'
    ]
    df['city_group'] = df['city'].where(df['city'].isin(top_cities), 'other_city')
    df = pd.get_dummies(df, columns=['city_group'], prefix='city')

    df['gen_group'] = df['generation_name']
    df = pd.get_dummies(df, columns=['gen_group'], prefix='gen')
    raw = pd.read_csv("data/02_intermediate/clean_dataset.csv", usecols=['price'])
    df['model_te'] = raw['price'].mean()
    return df

@st.cache_resource
def load_predictor(path: str) -> TabularPredictor:
    return TabularPredictor.load(path)

predictor = load_predictor("data/07_model_output/car_price_predictor_final")

template_cols = list(
    pd.read_csv("data/02_intermediate/features.csv", nrows=0).columns
)

st.title("🛻 Car Price Predictor")
mode = st.radio("Wybierz tryb:", ["Pojedyncze auto","Batch CSV"])

if mode == "Pojedyncze auto":
    st.subheader("Wprowadź dane samochodu:")
    c1, c2, c3 = st.columns(3)
    with c1:
        mark = st.selectbox("Marka", ["audi","bmw","mercedes-benz","other_mark"])
        model = st.text_input("Model","")
        year = st.number_input("Rok produkcji",1990,2025,2015)
    with c2:
        mileage = st.number_input("Przebieg [km]",2000,300000,50000)
        vol = st.number_input("Pojemność silnika [cc]",400,6600,2000)
        fuel = st.selectbox("Paliwo",["Gasoline","Diesel"])
    with c3:
        gen = st.text_input("Generation name","unknown")
        city = st.text_input("Miasto","other_city")
    if st.button("Oblicz cenę"):
        df_raw = pd.DataFrame([{
            'mark':mark,'model':model,'year':year,
            'mileage':mileage,'vol_engine':vol,
            'fuel':fuel,'generation_name':gen,'city':city
        }])
        df_c = inference_clean_data(df_raw)
        df_n = create_numerical_features(df_c, current_year=2025)
        df_s = scale_features(df_n)
        df_e = encode_inference(df_s)
        X_inf = df_e.drop(columns=[
            'year','mileage','mark','model','city','generation_name','fuel','fuel_type'
        ], errors='ignore').reindex(columns=template_cols, fill_value=0)
        price = predictor.predict(X_inf).iloc[0]
        st.success(f"Przewidywana cena: {price:,.0f} PLN")

else:
    st.subheader("Batch: wgraj features_df.csv")
    up = st.file_uploader("Wybierz plik CSV", type="csv")
    if up:
        df = pd.read_csv(up)
        df['predicted_price'] = predictor.predict(df)
        st.write("### Wyniki predykcji:")
        st.dataframe(df)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Pobierz CSV", csv,
                           file_name="predictions.csv",
                           mime="text/csv")