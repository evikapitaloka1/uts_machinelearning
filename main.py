# =============================
# Data Preprocessing & Transformation
# =============================

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # Ditambahkan untuk LDA
from imblearn.over_sampling import SMOTE

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
# Isi Age dengan median, Embarked dengan modus
df.fillna({
    'Age': df['Age'].median(),
    'Embarked': df['Embarked'].mode()[0]
}, inplace=True)

# Hapus kolom Cabin karena terlalu banyak missing value
if 'Cabin' in df.columns:
    df.drop(columns=['Cabin'], inplace=True)

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
df_minmax = pd.DataFrame(
    scaler_minmax.fit_transform(df_no_outlier[selected_cols]),
    columns=selected_cols
)
print("Min-Max Scaling (contoh 5 baris):")
print(df_minmax.head())

# --- Z-Score Standardization ---
scaler_zscore = StandardScaler()
df_zscore = pd.DataFrame(
    scaler_zscore.fit_transform(df_no_outlier[selected_cols]),
    columns=selected_cols
)
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
# 8. EKSTRAKSI FITUR DENGAN LDA (Linear Discriminant Analysis)
# ========================================================
print("\n=== 8. Menjalankan Ekstraksi Fitur dengan LDA ===")

# Pisahkan fitur (X) dan target (y) dari df_no_outlier agar konsisten
features_for_lda = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X_lda = df_no_outlier[features_for_lda]
y_lda = df_no_outlier['Survived']

# LDA bekerja lebih baik dengan data yang di-scaling (distandarisasi)
scaler_lda = StandardScaler()
X_lda_scaled = scaler_lda.fit_transform(X_lda)

# Inisialisasi dan menjalankan LDA
lda = LinearDiscriminantAnalysis(n_components=1)
X_lda_transformed = lda.fit_transform(X_lda_scaled, y_lda)

# Tampilkan hasilnya
print(f"Shape data fitur asli: {X_lda_scaled.shape}")
print(f"Shape data setelah transformasi LDA: {X_lda_transformed.shape}")
print("\nContoh 5 baris data setelah ekstraksi fitur dengan LDA (Komponen LD1):")
print(pd.DataFrame(X_lda_transformed, columns=['LD1']).head())

# Explained variance ratio menunjukkan seberapa banyak informasi yang ditangkap komponen baru
print(f"\nExplained Variance Ratio: {lda.explained_variance_ratio_[0]:.4f}")
print("Artinya, komponen baru (LD1) ini menangkap ~100% informasi yang memisahkan antar kelas.")

# Simpan hasil LDA ke file CSV baru
df_lda_transformed = pd.DataFrame(X_lda_transformed, columns=['LD1'])
# Gabungkan dengan kolom target
df_lda_transformed['Survived'] = y_lda.values
df_lda_transformed.to_csv("train_lda_transformed.csv", index=False)
print("\n-> File hasil ekstraksi fitur LDA telah disimpan ke 'train_lda_transformed.csv'.")


# ========================================================
# 9. PENYEIMBANGAN DATA DENGAN SMOTE & VISUALISASI
# ========================================================
print("\n=== 9. Menjalankan Penyeimbangan Data dengan SMOTE ===")

# Pisahkan fitur dan target untuk SMOTE dari df_encoded (seperti kode asli)
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