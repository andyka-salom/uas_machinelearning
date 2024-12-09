import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler

# Load dataset
file_path = '../uas_ml prak/skincare.csv'  # Lokasi file yang diunggah
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

# Deteksi Outlier menggunakan Z-Score
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
print("\nKolom numerik yang akan diperiksa outlier:", numeric_cols.tolist())

z_scores = np.abs(zscore(df[numeric_cols]))

# Menghapus baris dengan outlier
outliers_mask = (z_scores < 3).all(axis=1)
df = df[outliers_mask]
print(f"\nData setelah menghapus outlier: {df.shape[0]} baris")
print("============================================================")

# Normalisasi Data
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
print("\nData setelah normalisasi:")
print(df.head())
print("============================================================")

# Menyimpan data hasil preprocessing ke file baru
output_file = '../uas_ml prak/preprocessed_skincare.csv'
df.to_csv(output_file, index=False)
print(f"\nData hasil preprocessing disimpan ke: {output_file}")