#!/usr/bin/env python
# coding: utf-8

# # Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

# - Nama: Ahmad Rusdianto Andarina Syakbani
# - Email: dian.galleon@gmail.com
# - Id Dicoding: arusdian

# ## Persiapan

# ### Menyiapkan library yang dibutuhkan

# In[46]:


# 📦 General libraries
import os
import numpy as np
import pandas as pd

# 📊 Visualization
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# ⚙️ Sklearn - Data Preprocessing & Model Selection
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import precision_recall_curve

# 🧪 Sklearn - Metrics & Evaluation
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_curve, 
    auc
)

# 🌟 Boosting Libraries
from xgboost import XGBClassifier

# 🔧 Sampling & Optimization
from bayes_opt import BayesianOptimization

# Exporting the trained model
import joblib


# ### Menyiapkan data yang akan diguankan

# In[2]:


df = pd.read_csv('dataset/data.csv', sep=';', encoding='utf-8')
df.sample(5)


# In[3]:


image_path = os.path.join(os.getcwd(), "plots")
model_path = os.path.join(os.getcwd(), "models")
if not os.path.exists(image_path):
    os.makedirs(image_path)
    
if not os.path.exists(model_path):
    os.makedirs(model_path)


# ## Data Understanding

# In[4]:


eda_df = df.copy()


# In[5]:


# Type your code here
data_info = {
    "Jumlah Baris": eda_df.shape[0],
    "Jumlah Kolom": eda_df.shape[1],
    "Tipe Data": eda_df.dtypes.value_counts().to_dict(),
}

display(data_info)


# In[6]:


eda_df.info()


# In[7]:


eda_df.describe().T


# In[8]:


eda_df.sample(5).T


# In[9]:


mapping_dict = {
    "Marital_status": {
        1: "Single",
        2: "Married",
        3: "Widower",
        4: "Divorced",
        5: "Facto Union",
        6: "Legally Separated",
    },
    "Application_mode": {
        1: "1st phase - general",
        2: "Ordinance 612/93",
        5: "1st phase - Azores",
        7: "Other higher courses",
        10: "Ordinance 854-B/99",
        15: "International student",
        16: "1st phase - Madeira",
        17: "2nd phase - general",
        18: "3rd phase - general",
        26: "Different Plan",
        27: "Other Institution",
        39: "Over 23 years old",
        42: "Transfer",
        43: "Change of course",
        44: "Tech diploma",
        51: "Change of institution/course",
        53: "Short cycle diploma",
        57: "Change of institution/course (Intl)",
    },
    "Course": {
        33: "Biofuel Tech",
        171: "Animation & Multimedia",
        8014: "Social Service (Evening)",
        9003: "Agronomy",
        9070: "Comm. Design",
        9085: "Veterinary Nursing",
        9119: "Informatics Eng.",
        9130: "Equinculture",
        9147: "Management",
        9238: "Social Service",
        9254: "Tourism",
        9500: "Nursing",
        9556: "Oral Hygiene",
        9670: "Marketing Mgmt",
        9773: "Journalism & Comm.",
        9853: "Basic Education",
        9991: "Management (Evening)",
    },
    "Daytime_evening_attendance": {1: "Daytime", 0: "Evening"},
    "Previous_qualification": {
        1: "Secondary",
        2: "Bachelor",
        3: "Degree",
        4: "Master",
        5: "Doctorate",
        6: "Some Higher Edu.",
        9: "12th not completed",
        10: "11th not completed",
        12: "Other - 11th",
        14: "10th",
        15: "10th not completed",
        19: "Basic Edu 3rd Cycle",
        38: "Basic Edu 2nd Cycle",
        39: "Tech course",
        40: "Degree (1st cycle)",
        42: "Prof Higher Tech",
        43: "Master (2nd cycle)",
    },
    "Nacionality": {
        1: "Portuguese",
        2: "German",
        6: "Spanish",
        11: "Italian",
        13: "Dutch",
        14: "English",
        17: "Lithuanian",
        21: "Angolan",
        22: "Cape Verdean",
        24: "Guinean",
        25: "Mozambican",
        26: "Santomean",
        32: "Turkish",
        41: "Brazilian",
        62: "Romanian",
        100: "Moldova",
        101: "Mexican",
        103: "Ukrainian",
        105: "Russian",
        108: "Cuban",
        109: "Colombian",
    },
    "Displaced": {1: "Yes", 0: "No"},
    "Educational_special_needs": {1: "Yes", 0: "No"},
    "Debtor": {1: "Yes", 0: "No"},
    "Tuition_fees_up_to_date": {1: "Yes", 0: "No"},
    "Gender": {1: "Male", 0: "Female"},
    "Scholarship_holder": {1: "Yes", 0: "No"},
    "International": {1: "Yes", 0: "No"},
}

for col, mapping in mapping_dict.items():
    if col in eda_df.columns:
        eda_df[col] = eda_df[col].map(mapping)


# ### Exploratory Data Analysis : Data Cleaning

# In[10]:


# Cek missing values
missing = df.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(ascending=False)


# In[11]:


# Cek data duplikat
print(f"Jumlah data duplikat: {eda_df.duplicated().sum()}")


# In[12]:


print(df.columns.tolist())


# In[13]:


# Cek label unik tiap fitur kategorikal
for col in eda_df.select_dtypes(include="object").columns:
    print(f"{col}: {eda_df[col].value_counts().to_dict()}")


# ### Exploratory Data Analysis : Univariate Analysis

# In[14]:


numerical_cols = eda_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = eda_df.select_dtypes(include="object").columns.tolist()


# In[15]:


numerical_cols = eda_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
n = len(numerical_cols)
rows = int(np.ceil(n / 3))
fig, axes = plt.subplots(rows, 3, figsize=(18, rows * 4))
axes = axes.flatten()

for i, col in enumerate(numerical_cols):
    sns.histplot(eda_df[col], kde=True, ax=axes[i], bins=30)
    axes[i].set_title(f"Distribusi {col}")

# Hapus subplot kosong jika ada
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig(os.path.join(image_path, "numerical_distributions.png"))
plt.show()


# In[16]:


n = len(categorical_cols)
rows = int(np.ceil(n / 3))
fig, axes = plt.subplots(rows, 3, figsize=(18, rows * 4))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    sns.countplot(
        data=eda_df, x=col, order=eda_df[col].value_counts().index, ax=axes[i]
    )
    axes[i].set_title(f"Distribusi {col}")
    axes[i].tick_params(axis="x", rotation=45)

# Hapus subplot kosong jika ada
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig(os.path.join(image_path, "categorical_distributions.png"))
plt.show()


# ### Exploratory Data Analysis : Bivariate Analysis

# In[17]:


# Daftar kolom kategorikal yang sudah dimapping
categorical_cols = [
    "Marital_status",
    "Application_mode",
    "Course",
    "Daytime_evening_attendance",
    "Previous_qualification",
    "Nacionality",
    "Displaced",
    "Educational_special_needs",
    "Debtor",
    "Tuition_fees_up_to_date",
    "Gender",
    "Scholarship_holder",
    "International",
]

# Setup subplot grid
n_cols = 3
n_rows = int(np.ceil(len(categorical_cols) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
axes = axes.flatten()

# Plot tiap kolom dalam subplot
for i, col in enumerate(categorical_cols):
    if col in df.columns:
        sns.countplot(data=df, x=col, hue="Status", ax=axes[i])
        axes[i].set_title(f"{col} vs Status")
        axes[i].tick_params(axis="x", rotation=45)
    else:
        axes[i].axis("off")  # Kosongkan subplot kalau tidak ada kolom

# Kosongkan subplot jika tidak cukup kolom
for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(image_path, "categorical_vs_status.png"))
plt.show()


# In[18]:


# Pilih kolom numerikal utama
numerical_cols = [
    "Age_at_enrollment",
    "Admission_grade",
    "Curricular_units_1st_sem_enrolled",
    "Curricular_units_1st_sem_approved",
    "Curricular_units_1st_sem_grade",
    "Curricular_units_2nd_sem_enrolled",
    "Curricular_units_2nd_sem_approved",
    "Curricular_units_2nd_sem_grade",
]

# Setup subplot grid
n_cols = 2
n_rows = int(np.ceil(len(numerical_cols) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
axes = axes.flatten()

# Plot boxplot untuk tiap fitur numerikal
for i, col in enumerate(numerical_cols):
    if col in df.columns:
        sns.boxplot(data=df, x="Status", y=col, ax=axes[i])
        axes[i].set_title(f"{col} vs Status")
        axes[i].set_xlabel("Status")
        axes[i].set_ylabel(col)
    else:
        axes[i].axis("off")

# Kosongkan subplot jika tidak cukup fitur
for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(image_path, "numerical_vs_status.png"))
plt.show()


# ### Exploratory Data Analysis : Multivariate Analysis

# In[19]:


# Korelasi numerikal
plt.figure(figsize=(14, 10))
corr_matrix = df[numerical_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap antar Fitur Numerikal")
plt.savefig(os.path.join(image_path, "numerical_correlation_heatmap.png"))
plt.show()


# In[20]:


# Pilih subset numerikal yang paling relevan
subset = [
    "Admission_grade",
    "Age_at_enrollment",
    "Curricular_units_1st_sem_approved",
    "Curricular_units_2nd_sem_approved",
    "Curricular_units_1st_sem_grade",
    "Curricular_units_2nd_sem_grade",
    "Status",
]

# Hanya ambil sampel kecil jika dataset besar
sns.pairplot(df[subset].sample(500), hue="Status", diag_kind="kde")
plt.suptitle("Pairplot Multivariat Terhadap Status", y=1.02)
plt.savefig(os.path.join(image_path, "pairplot_multivariate_status.png"))
plt.show()


# ## Data Preparation / Preprocessing

# In[21]:


preprocessing_df = eda_df.copy()


# In[22]:


preprocessing_df["target"] = preprocessing_df["Status"].map(
    {"Dropout": 1, "Graduate": 0, "Enrolled": 0}
)  # Fokus dropout

preprocessing_df.drop(columns=["Status"], inplace=True)


# In[23]:


# 1. Split data sebelum encoding/scaling agar tidak terjadi data leakage
X = preprocessing_df.drop(columns="target")
y = preprocessing_df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# In[24]:


# Daftar fitur kategorikal hasil mapping
categorical_features = [
    "Gender",
    "Debtor",
    "Displaced",
    "Scholarship_holder",
    "International",
    "Tuition_fees_up_to_date",
    "Educational_special_needs",
    "Daytime_evening_attendance",
    "Marital_status",
    "Application_mode",
    "Course",
    "Previous_qualification",
    "Nacionality",
]

# Fitur numerikal dari observasi sebelumnya
numerical_features = [
    "Age_at_enrollment",
    "Admission_grade",
    "Curricular_units_1st_sem_enrolled",
    "Curricular_units_1st_sem_approved",
    "Curricular_units_1st_sem_grade",
    "Curricular_units_2nd_sem_enrolled",
    "Curricular_units_2nd_sem_approved",
    "Curricular_units_2nd_sem_grade",
]


# In[25]:


def encoding(features, df, df_test=None):

    if df_test is not None:
        df = df.copy()
        df_test = df_test.copy()
        for feature in features:
            encoder = LabelEncoder()
            combined = pd.concat([df[feature], df_test[feature]], axis=0)
            encoder.fit(combined)

            df[feature] = encoder.transform(df[feature])
            df_test[feature] = encoder.transform(df_test[feature])

            joblib.dump(encoder, os.path.join(model_path, f"encoder_{feature}.joblib"))
        return df, df_test
    else:
        df = df.copy()
        for feature in features:
            encoder = LabelEncoder()
            encoder.fit(df[feature])
            df[feature] = encoder.transform(df[feature])
            joblib.dump(encoder, os.path.join(model_path, f"encoder_{feature}.joblib"))
        return df


# In[30]:


def scaling(features, df, df_test=None):

    if df_test is not None:
        df = df.copy()
        df_test = df_test.copy()
        for feature in features:
            scaler = StandardScaler()
            X = df[[feature]]
            scaler.fit(X)
            df[feature] = scaler.transform(X)
            joblib.dump(scaler, os.path.join(model_path, f"scaler_{feature}.joblib"))

            X_test = df_test[[feature]]
            df_test[feature] = scaler.transform(X_test)
        return df, df_test
    else:
        df = df.copy()
        for feature in features:
            scaler = StandardScaler()
            X = df[[feature]]
            scaler.fit(X)
            df[feature] = scaler.transform(X)
            joblib.dump(scaler, os.path.join(model_path, f"scaler_{feature}.joblib"))
        return df


# In[31]:


# 1. Encoding fitur kategorikal
X_train_enc, X_test_enc = encoding(categorical_features, X_train, X_test)

# 2. Scaling fitur numerikal
X_train_scaled, X_test_scaled = scaling(numerical_features, X_train_enc, X_test_enc)


# In[32]:


X_train_scaled["target"] = y_train.values
X_test_scaled["target"] = y_test.values

X_train_scaled.to_csv(os.path.join(model_path, "preprocessed_train.csv"), index=False)
X_test_scaled.to_csv(os.path.join(model_path, "preprocessed_test.csv"), index=False)


# In[33]:


X_train_final = X_train_scaled.drop(columns=["target"])
y_train_final = X_train_scaled["target"]

X_test_final = X_test_scaled.drop(columns=["target"])
y_test_final = X_test_scaled["target"]


# ## Modeling

# In[36]:


# Hitung rasio untuk scale_pos_weight
negative = y_train_final.value_counts()[0]
positive = y_train_final.value_counts()[1]
scale_pos_weight = negative / positive
print(f"scale_pos_weight = {scale_pos_weight:.2f}")


# In[38]:


def xgb_cv(max_depth, learning_rate, n_estimators, gamma, subsample, colsample_bytree):
    model = XGBClassifier(
        max_depth=int(max_depth),
        learning_rate=learning_rate,
        n_estimators=int(n_estimators),
        gamma=gamma,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )
    scores = cross_val_score(
        model, X_train_final, y_train_final, scoring="roc_auc", cv=3
    )
    return scores.mean()

xgb_bo = BayesianOptimization(
    f=xgb_cv,
    pbounds={
        "max_depth": (3, 10),
        "learning_rate": (0.01, 0.3),
        "n_estimators": (50, 300),
        "gamma": (0, 5),
        "subsample": (0.6, 1),
        "colsample_bytree": (0.6, 1),
    },
    random_state=42,
    verbose=2,
)

xgb_bo.maximize(init_points=5, n_iter=10)


# In[39]:


best_params = xgb_bo.max["params"]
best_model = XGBClassifier(
    max_depth=int(best_params["max_depth"]),
    learning_rate=best_params["learning_rate"],
    n_estimators=int(best_params["n_estimators"]),
    gamma=best_params["gamma"],
    subsample=best_params["subsample"],
    colsample_bytree=best_params["colsample_bytree"],
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
)

best_model.fit(X_train_final, y_train_final)


# ## Evaluation

# In[54]:


# Prediksi
y_pred = best_model.predict(X_test_final)
y_proba = best_model.predict_proba(X_test_final)[:, 1]

cm = confusion_matrix(y_test_final, y_pred)
labels = ["Not Dropout", "Dropout"]

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(image_path, "confusion_matrix.png"))
plt.show()


# In[53]:


# Buat DataFrame dari classification report
report_dict = classification_report(y_test_final, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

# Plot sebagai tabel
plt.figure(figsize=(8, 4))
sns.heatmap(report_df.iloc[:-1, :-1], annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Classification Report")
plt.tight_layout()
plt.savefig(os.path.join(image_path, "classification_report.png"))
plt.show()


# In[55]:


fpr, tpr, thresholds = roc_curve(y_test_final, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(image_path, "roc_curve.png"))
plt.show()


# In[56]:


importances = best_model.feature_importances_
feature_names = X_train_final.columns

# Plot
plt.figure(figsize=(8, 6))
sorted_idx = np.argsort(importances)[::-1][:15]
plt.barh(range(len(sorted_idx)), importances[sorted_idx])
plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Top 15 Feature Importances (XGBoost)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(image_path, "feature_importances.png"))
plt.show()


# In[57]:


from sklearn.metrics import precision_recall_curve

precision, recall, thresholds_pr = precision_recall_curve(y_test_final, y_proba)

plt.figure(figsize=(6, 5))
plt.plot(recall, precision, marker=".")
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(image_path, "precision_recall_curve.png"))
plt.show()


# In[58]:


joblib.dump(best_model, os.path.join(model_path, "xgb_dropout_model.joblib"))


# ## Database Connection

# In[ ]:





# In[59]:


get_ipython().system('pip install pipreqs')


# In[ ]:




