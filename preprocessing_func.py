import joblib
import pandas as pd
import os
import numpy as np

# Path ke folder model
model_path = "models/"

# Semua fitur yang digunakan saat training, dalam urutan yang benar
all_features = [
    "Marital_status",
    "Application_mode",
    "Application_order",
    "Course",
    "Daytime_evening_attendance",
    "Previous_qualification",
    "Previous_qualification_grade",
    "Nacionality",
    "Mothers_qualification",
    "Fathers_qualification",
    "Mothers_occupation",
    "Fathers_occupation",
    "Admission_grade",
    "Displaced",
    "Educational_special_needs",
    "Debtor",
    "Tuition_fees_up_to_date",
    "Gender",
    "Scholarship_holder",
    "Age_at_enrollment",
    "International",
    "Curricular_units_1st_sem_credited",
    "Curricular_units_1st_sem_enrolled",
    "Curricular_units_1st_sem_evaluations",
    "Curricular_units_1st_sem_approved",
    "Curricular_units_1st_sem_grade",
    "Curricular_units_1st_sem_without_evaluations",
    "Curricular_units_2nd_sem_credited",
    "Curricular_units_2nd_sem_enrolled",
    "Curricular_units_2nd_sem_evaluations",
    "Curricular_units_2nd_sem_approved",
    "Curricular_units_2nd_sem_grade",
    "Curricular_units_2nd_sem_without_evaluations",
    "Unemployment_rate",
    "Inflation_rate",
    "GDP",
]


def preprocess_input_data(df):
    df = df.copy()

    # Tambahkan semua kolom yang hilang dengan default 0
    for col in all_features:
        if col not in df.columns:
            df[col] = 0

    # ENCODING: Label encode semua kolom object
    for col in df.columns:
        if df[col].dtype == "object":
            enc_path = os.path.join(model_path, f"encoder_{col}.joblib")
            if os.path.exists(enc_path):
                encoder = joblib.load(enc_path)
                df[col] = encoder.transform(df[col])
            else:
                # Fallback jika tidak ditemukan: encode dengan kode kategori
                df[col] = df[col].astype("category").cat.codes

    # SCALING: Terapkan scaler untuk kolom numerik jika tersedia
    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:
            scaler_path = os.path.join(model_path, f"scaler_{col}.joblib")
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                df[col] = scaler.transform(df[[col]])

    # Pastikan semua kolom ada dan urut
    df = df[all_features]

    # Pastikan semua kolom bertipe float (aman untuk XGBoost)
    df = df.astype(float)

    return df
