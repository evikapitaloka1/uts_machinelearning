# =============================
# Data Preprocessing & Transformation
# =============================

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.over_sampling import SMOTE  # Ditambahkan untuk SMOTE

le = LabelEncoder()


# =============================
# 1. Load Dataset
# =============================
df = pd.read_csv("train.csv")
print("=== Dataset Awal ===")
print(df.head())
print("\nInfo Dataset:")
df.info()

# =============================
# 2. Cek Missing Value
# =============================
print("\n=== Mengecek Missing Value ===")
print(df.isnull().sum())

# =============================
# 3. Handling Missing Value
# =============================
# Menggunakan metode yang lebih baik dari dropna()
# Isi 'Age' dengan median
df['Age'].fillna(df['Age'].median(), inplace=True)
# Isi 'Embarked' dengan modus
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
# Hapus kolom 'Cabin' karena terlalu banyak missing value
df.drop('Cabin', axis=1, inplace=True)

print("\nDataset setelah menangani missing value:")
print(df.isnull().sum())


# =============================
# 4. Encoding Variabel Kategorikal
# =============================
print("\n=== 4. Menjalankan Encoding Variabel Kategorikal ===")
df_encoded = df.copy()
df_encoded['Sex'] = le.fit_transform(df_encoded['Sex'])
df_encoded['Embarked'] = le.fit_transform(df_encoded['Embarked'])

print("Contoh data setelah encoding:")
print(df_encoded.head())

# =============================
# 5. Deteksi & Handling Outlier (Z-Score)
# =============================
print("\n=== 5. Menjalankan Deteksi & Handling Outlier ===")
numerical_cols = ['Age', 'SibSp', 'Parch', 'Fare']
z_scores = np.abs(stats.zscore(df_encoded[numerical_cols]))
df_no_outlier = df_encoded[(z_scores < 3).all(axis=1)]

print(f"Jumlah baris sebelum menghapus outlier: {len(df_encoded)}")
print(f"Jumlah baris setelah menghapus outlier: {len(df_no_outlier)}")


# =============================
# 6. Transformasi Data
# =============================
print("\n=== 6. Menjalankan Transformasi Data ===")
# --- Min-Max Scaling ---
scaler_minmax = preprocessing.MinMaxScaler()
selected_cols = ['Age', 'Fare']
df_minmax = pd.DataFrame(scaler_minmax.fit_transform(df_no_outlier[selected_cols]),
                         columns=selected_cols)
print("Min-Max Scaling (contoh 5 baris):")
print(df_minmax.head())

# --- Z-Score Standardization ---
scaler_zscore = StandardScaler()
df_zscore = pd.DataFrame(scaler_zscore.fit_transform(df_no_outlier[selected_cols]),
                         columns=selected_cols)
print("\nZ-Score Standardization (contoh 5 baris):")
print(df_zscore.head())

# =============================
# 7. Simpan hasil
# =============================
df_encoded.to_csv("train_encoded.csv", index=False)
df_no_outlier.to_csv("train_cleaned_no_outlier.csv", index=False)
df_minmax.to_csv("train_minmax.csv", index=False)
df_zscore.to_csv("train_zscore.csv", index=False)
print("\nFile-file hasil preprocessing telah disimpan.")

# ========================================================
# 8. SELEKSI FITUR DENGAN CHI-SQUARE
# ========================================================
print("\n=== 8. Menjalankan Seleksi Fitur Chi-Square ===")

# Pisahkan fitur (X) dan target (y) dari df_encoded
features_for_selection = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Fare']
X_fs = df_encoded[features_for_selection]
y_fs = df_encoded['Survived']

# Inisialisasi dan jalankan SelectKBest untuk memilih 4 fitur terbaik
chi2_selector = SelectKBest(score_func=chi2, k=4)
X_kbest = chi2_selector.fit_transform(X_fs, y_fs)

# Tampilkan hasilnya
selected_features_mask = chi2_selector.get_support()
selected_features = X_fs.columns[selected_features_mask]
scores = chi2_selector.scores_[selected_features_mask]

print("Fitur terbaik yang dipilih berdasarkan Chi-Square:")
for feature, score in zip(selected_features, scores):
    print(f"- {feature}: {score:.2f}")

# ========================================================
# 9. PENYEIMBANGAN DATA DENGAN SMOTE & VISUALISASI
# ========================================================
print("\n=== 9. Menjalankan Penyeimbangan Data dengan SMOTE ===")

# Pisahkan fitur dan target untuk SMOTE
X_smote = df_encoded.drop(['Survived', 'PassengerId', 'Name', 'Ticket'], axis=1)
y_smote = df_encoded['Survived']

# --- Visualisasi Sebelum SMOTE ---
print("\n-> Distribusi kelas sebelum SMOTE:")
print(y_smote.value_counts())
plt.figure(figsize=(8, 6))
y_smote.value_counts().plot(kind='bar', color=['#3498db', '#e74c3c'])
plt.title('Distribusi Kelas Sebelum SMOTE (Tidak Seimbang)', fontsize=16)
plt.xlabel('Status Selamat', fontsize=12)
plt.ylabel('Jumlah Penumpang', fontsize=12)
plt.xticks(ticks=[0, 1], labels=['Tidak Selamat', 'Selamat'], rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('distribusi_sebelum_smote.png')
print("   -> Plot disimpan ke 'distribusi_sebelum_smote.png'")

# --- Terapkan SMOTE ---
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_smote, y_smote)
print("\n-> SMOTE telah diterapkan.")

# --- Visualisasi Setelah SMOTE ---
print("\n-> Distribusi kelas setelah SMOTE:")
print(y_resampled.value_counts())
plt.figure(figsize=(8, 6))
y_resampled.value_counts().plot(kind='bar', color=['#2ecc71', '#f39c12'])
plt.title('Distribusi Kelas Setelah SMOTE (Seimbang)', fontsize=16)
plt.xlabel('Status Selamat', fontsize=12)
plt.ylabel('Jumlah Penumpang', fontsize=12)
plt.xticks(ticks=[0, 1], labels=['Tidak Selamat', 'Selamat'], rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('distribusi_setelah_smote.png')
print("   -> Plot disimpan ke 'distribusi_setelah_smote.png'")

# --- Simpan Dataset yang Seimbang ---
df_balanced = pd.DataFrame(X_resampled, columns=X_smote.columns)
df_balanced['Survived'] = y_resampled
df_balanced.to_csv('train_balanced_smote.csv', index=False)
print("\n-> Dataset yang seimbang telah disimpan ke 'train_balanced_smote.csv'")

print("\n\n=== SEMUA PROSES SELESAI ===")