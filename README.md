# Car Price Predictor

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)  
[![Built with Streamlit](https://img.shields.io/badge/built_with-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io)

## Overview

Przewidywanie cen samochodów na podstawie danych z Kaggle, z wykorzystaniem frameworka Kedro do przygotowania danych, AutoGluon do AutoML oraz front-endu Streamlit do interaktywnej prezentacji wyników.

W projekcie znajdziesz:

* **Data Preparation** (`src/carprices/pipelines/data_preparation/`): czyszczenie, inżynieria cech i zapisywanie artefaktów.  
* **Model Training** (`src/carprices/pipelines/autogluon_pipeline/`): trening modeli AutoGluon, wybór najlepszego i jego ewaluacja.  
* **Application** (`app.py`): interfejs użytkownika w Streamlit do predykcji pojedynczych aut.  
* **Dockerfile**: konteneryzacja aplikacji Streamlit i zależności.  

## Prerequisites

Wymagania dotyczące uruchomienia kodu:

- **Python** 3.10, 3.11 or 3.12  
- **Docker** (if you want to build/run the container)  
- **Kedro** ≥ 0.19.12  
- **Streamlit**, **AutoGluon**, **scikit-learn**, etc.  

## Quickstart (lokalnie)

1. **Zainstaluj zależności**  
   ```bash
   pip install -r requirements.txt

## Quickstart (lokalnie)

1. **Zainstaluj zależności**

   ```bash
   pip install -r requirements.txt
   ```

2. **Uruchom Kedro pipeline**

   ```bash
   kedro run --pipeline data_preparation
   kedro run --pipeline autogluon_pipeline
   ```

3. **Otwórz aplikację Streamlit**

   ```bash
   streamlit run app.py
   ```

4. Aplikacja dostępna pod adresem `http://localhost:8501`.

## Budowanie i uruchomienie w Dockerze

1. **Zbuduj obraz**

   ```bash
   docker build -t car-price-predictor:latest .
   ```
2. **Uruchom kontener lokalnie**

   ```bash
   docker run -p 8501:8501 car-price-predictor:latest
   ```
3. Aplikacja będzie dostępna pod `http://localhost:8501`.

---

**Link do wdrożonej aplikacji:**
 https://car-price-app-egf6ccgye3f5ehcp.polandcentral-01.azurewebsites.net/