import pandas as pd
import numpy as np
from scipy.stats import zscore
import os

# Load dataset
file_path = '../uas_ml prak/uas_machinelearning/skincare.csv'
df = pd.read_csv(file_path)

# Menampilkan informasi awal dataset
print("Informasi Dataset:")
print(df.info())
print("\nData:")
print(df.head())
print("============================================================")

# Pengecekan Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Menghapus baris dengan nilai yang hilang
df.dropna(inplace=True)

print("\nData setelah menghapus baris dengan missing values:")
print(df.info())
print("============================================================")

# deteksi outlier hanya untuk kolom 'price' dan 'rank'
columns_to_check = ['Price', 'Rank']

# Validasi tipe data sebelum melakukan Z-score
for col in columns_to_check:
    if not np.issubdtype(df[col].dtype, np.number):
        print(f"Koleksi {col} bukan tipe numerik. Periksa dataset.")
        exit()

# Menghitung Z-score untuk mendeteksi outlier
z_scores = zscore(df[columns_to_check].values, axis=0)  # Menggunakan .values untuk memastikan format array NumPy
outliers_per_column = {}

# Debugging untuk memeriksa bentuk z_scores
print(f"Shape dari z_scores: {z_scores.shape}")
print("============================================================")

for idx, col in enumerate(columns_to_check):
    # Debugging untuk memeriksa panjang indeks
    print(f"Memeriksa indeks untuk kolom: {col}, panjang kolom: {len(df[col])}, panjang z_scores: {len(z_scores[:, idx])}")

    # Menghindari kesalahan dengan memastikan panjang array cocok
    if len(z_scores[:, idx]) == len(df[col]):
        outliers = df[col][z_scores[:, idx] > 3].tolist()
        outliers_per_column[col] = outliers
    else:
        print(f"Kesalahan panjang array: {len(z_scores[:, idx])} dan {len(df[col])}")

# Mengganti outlier dengan nilai rata-rata
df_cleaned = df.copy()
for col in columns_to_check:
    if len(outliers_per_column[col]) > 0:
        # Hitung nilai rata-rata dari kolom tanpa outlier
        mean_value = df_cleaned[~df_cleaned[col].isin(outliers_per_column[col])][col].mean()
        df_cleaned[col] = df_cleaned[col].apply(lambda x: mean_value if x in outliers_per_column[col] else x)

print("\nData setelah mengganti outlier dengan rata-rata:")
print(df_cleaned.head())
print("============================================================")

# Simpan hasil preprocessing ke file baru
# output_file = '../uas_ml prak/uas_machinelearning/preprocessedbaru_skincare.csv'
# df_cleaned.to_csv(output_file, index=False)
# print(f"\nData hasil preprocessing disimpan ke: {output_file}")
