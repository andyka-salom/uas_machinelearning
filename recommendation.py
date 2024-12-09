import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import streamlit as st
import numpy as np
from scipy.stats import zscore

# Set page configuration (HARUS menjadi perintah pertama)
st.set_page_config(page_title="Skincare Products Recommendation System", layout="wide")

# Read the dataset
df = pd.read_csv('../UAS_ML PRAK/uas_machinelearning/prepocessed_skincare.csv')  # Assuming the dataset is available

# Simulate 'Liked' column for user feedback (0 = not liked, 1 = liked)
df['Liked'] = np.random.choice([0, 1], size=len(df))

# Check for missing values in relevant columns
st.sidebar.title("Data Insights")
missing_values = df.isna().sum()
st.sidebar.write("Jumlah nilai yang hilang per kolom:")
st.sidebar.write(missing_values)

# Detect and handle outliers using Z-Score
def detect_outliers_zscore(data, column, threshold=3):
    z_scores = zscore(data[column])
    outliers = data[np.abs(z_scores) > threshold]
    return outliers, z_scores

# Function to handle outliers
def handle_outliers(data, column, method='remove', threshold=3):
    z_scores = zscore(data[column])
    if method == 'remove':
        # Remove outliers
        data = data[np.abs(z_scores) <= threshold]
    elif method == 'replace':
        # Replace outliers with the median
        median_value = data[column].median()
        data.loc[np.abs(z_scores) > threshold, column] = median_value
    return data

st.sidebar.write("\n📊 Deteksi dan Penanganan Outliers:")

# Detect outliers for 'Price'
outliers_price, z_scores_price = detect_outliers_zscore(df, 'Price')
st.sidebar.write(f"Outliers in 'Price': {len(outliers_price)}")
st.sidebar.write(outliers_price[['Name', 'Price']])

# Detect outliers for 'Rank'
outliers_rank, z_scores_rank = detect_outliers_zscore(df, 'Rank')
st.sidebar.write(f"Outliers in 'Rank': {len(outliers_rank)}")
st.sidebar.write(outliers_rank[['Name', 'Rank']])

# Handle outliers (user can choose the method in sidebar)
handle_method = st.sidebar.selectbox("Pilih metode penanganan outliers:", ["remove", "replace"])
df_cleaned = handle_outliers(df, 'Price', method=handle_method)
df_cleaned = handle_outliers(df_cleaned, 'Rank', method=handle_method)

# Inform user about the result of outlier handling
st.sidebar.write("📊 Data setelah penanganan outliers:")
st.sidebar.write(f"Jumlah baris data: {len(df_cleaned)}")

# Create a feature column for ingredients
df_cleaned['features'] = df_cleaned['Ingredients']

# Create a TF-IDF vectorizer and fit on the 'features' column
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_cleaned['features'])

# Compute cosine similarity between all products
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create a series to map product names to their indices
indices = pd.Series(df_cleaned.index, index=df_cleaned['Name']).drop_duplicates()

# Function to recommend products based on filters and ingredient similarity
def recommend_cosmetics(skin_type, label_filter, rank_filter, brand_filter, price_range, ingredient_input=None, num_recommendations=10):
    # Filter products based on skin type and other filters
    recommended_products = df_cleaned[df_cleaned[skin_type] == 1]
    
    if label_filter != 'All':
        recommended_products = recommended_products[recommended_products['Label'] == label_filter]
    
    recommended_products = recommended_products[
        (recommended_products['Rank'] >= rank_filter[0]) & 
        (recommended_products['Rank'] <= rank_filter[1])
    ]
    
    if brand_filter != 'All':
        recommended_products = recommended_products[recommended_products['Brand'] == brand_filter]
    
    recommended_products = recommended_products[
        (recommended_products['Price'] >= price_range[0]) & 
        (recommended_products['Price'] <= price_range[1])
    ]

    # If ingredient input is provided, recommend products based on ingredient similarity
    if ingredient_input:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(df_cleaned['Ingredients'])
        input_vec = vectorizer.transform([ingredient_input])
        cosine_similarities = cosine_similarity(input_vec, tfidf_matrix).flatten()
        recommended_indices = cosine_similarities.argsort()[-num_recommendations:][::-1]
        ingredient_recommendations = df_cleaned.iloc[recommended_indices]
        recommended_products = recommended_products[recommended_products.index.isin(ingredient_recommendations.index)]
    
    return recommended_products.sort_values(by=['Rank']).head(num_recommendations)

# Evaluation Metrics
def evaluate_recommendations(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return accuracy, precision, recall, f1

# Main function
def main():
    # Header with styling
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 32px;
            font-weight: bold;
            text-align: center;
            color: #4CAF50;
        }
        .sub-title {
            font-size: 20px;
            text-align: center;
            color: #888888;
        }
        .footer {
            text-align: center;
            color: #888888;
            font-size: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='main-title'>🌿 Skincare Products Recommendation System 🌿</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Find the perfect skincare products tailored to your needs</div>", unsafe_allow_html=True)

    # Layout: Input Filters
    st.markdown("---")
    st.subheader("🔍 Search Filters")
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        skin_type = st.selectbox(
            'Select your skin type:',
            ('Combination', 'Dry', 'Normal', 'Oily', 'Sensitive')
        )

    unique_labels = df_cleaned['Label'].unique().tolist()
    unique_labels.insert(0, 'All')

    with col2:
        label_filter = st.selectbox(
            'Filter by label (optional):',
            unique_labels
        )

    with col1:
        rank_filter = st.slider(
            'Select rank range:',
            min_value=int(df_cleaned['Rank'].min()),
            max_value=int(df_cleaned['Rank'].max()),
            value=(int(df_cleaned['Rank'].min()), int(df_cleaned['Rank'].max()))
        )

    unique_brands = df_cleaned['Brand'].unique().tolist()
    unique_brands.insert(0, 'All')

    with col2:
        brand_filter = st.selectbox(
            'Filter by brand (optional):',
            unique_brands
        )

    with col3:
        price_range = st.slider(
            'Select price range:',
            min_value=float(df_cleaned['Price'].min()),
            max_value=float(df_cleaned['Price'].max()),
            value=(float(df_cleaned['Price'].min()), float(df_cleaned['Price'].max()))
        )

    st.markdown("---")

    # Ingredient input field
    st.subheader("🧴 Filter by Ingredients (Optional)")
    ingredient_input = st.text_area("Enter ingredients (comma-separated):", "")

    # Button to find similar products
    if st.button('🔍 Find Similar Products!'):
        # Get recommended products
        top_recommended_products = recommend_cosmetics(
            skin_type, label_filter, rank_filter, brand_filter, price_range, ingredient_input
        )

        # Display recommendations
        if not top_recommended_products.empty:
            st.markdown("### ✨ Recommended Products")
            st.dataframe(
                top_recommended_products[['Label', 'Brand', 'Name', 'Price', 'Rank', 'Ingredients']],
                height=400
            )

            # Simulate user feedback for evaluation (this part should be replaced with actual feedback data)
            y_true = top_recommended_products['Liked'].values
            y_pred = np.random.choice([0, 1], size=len(y_true))  # Simulate random predictions

            # Evaluate the recommendations
            accuracy, precision, recall, f1 = evaluate_recommendations(y_true, y_pred)

            # Display evaluation metrics
            st.markdown("### 🧮 Evaluation Metrics")
            st.write(f'Accuracy: {accuracy:.2f}')
            st.write(f'Precision: {precision:.2f}')
            st.write(f'Recall: {recall:.2f}')
            st.write(f'F1 Score: {f1:.2f}')
        else:
            st.warning("No products found! Try adjusting the filters or input.")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div class='footer'>© 2024 Skincare Recommendation System | Built with ❤️ by Informatics Engineer</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
