# ==============================
# Data Preprocessing & Transformation
# ==============================

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

# ==============================
# 1. Load Dataset
# ==============================
df = pd.read_csv("train.csv")
print("=== Dataset Awal ===")
print(df.head())
print("\nInfo Dataset:")
print(df.info())

# ==============================
# 2. Cek Missing Value
# ==============================
print("\n=== Mengecek Missing Value ===")
print(df.isnull().sum())

# ==============================
# 3. Handling Missing Value
# ==============================
df_cleaned = df.dropna()
print("\nDataset setelah menghapus missing value:")
print(df_cleaned.isnull().sum())

# ==============================
# 4. Deteksi Outlier (Z-Score)
# ==============================
numerical_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
print("\n=== Kolom Numerik Tersedia ===")
print(numerical_cols)

z_scores = np.abs(stats.zscore(df_cleaned[numerical_cols]))
outliers = df_cleaned[(z_scores > 3).any(axis=1)]
print(f"\nJumlah outlier terdeteksi: {len(outliers)}")

plt.figure(figsize=(10,5))
df_cleaned[numerical_cols].boxplot()
plt.title("Boxplot Sebelum Handling Outlier")
plt.show()

# ==============================
# 5. Handling Outlier (hapus data outlier)
# ==============================
df_no_outlier = df_cleaned[(z_scores <= 3).all(axis=1)]
print(f"\nDataset setelah menghapus outlier: {df_no_outlier.shape}")

plt.figure(figsize=(10,5))
df_no_outlier[numerical_cols].boxplot()
plt.title("Boxplot Sesudah Handling Outlier")
plt.show()

# ==============================
# 6. Data Transformation
# ==============================
print("\n=== Transformasi Data ===")

# --- Pilih kolom numerik untuk dinormalisasi ---
selected_cols = ["Age", "Fare", "SibSp", "Parch"]
print(f"\nKolom yang dipilih untuk normalisasi: {selected_cols}")

# --- Simple Feature Scaling ---
dfsfs = df_no_outlier[selected_cols] / df_no_outlier[selected_cols].max()
print("\nSimple Feature Scaling (contoh 5 baris):")
print(dfsfs.head())

# --- Min-Max Normalization ---
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(df_no_outlier[selected_cols])
df_minmax = pd.DataFrame(np_scaled, columns=selected_cols)
print("\nMin-Max Normalization (contoh 5 baris):")
print(df_minmax.head())

# --- Z-Score Standardization ---
scaler = StandardScaler()
df_zscore = pd.DataFrame(scaler.fit_transform(df_no_outlier[selected_cols]),
                         columns=selected_cols)
print("\nZ-Score Standardization (contoh 5 baris):")
print(df_zscore.head())

# ==============================
# 7. Simpan hasil
# ==============================
df_no_outlier.to_csv("train_cleaned.csv", index=False)
df_minmax.to_csv("train_minmax.csv", index=False)
df_zscore.to_csv("train_zscore.csv", index=False)

print("\n=== Semua proses selesai. File hasil tersimpan ===")
