import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import streamlit as st
import numpy as np
# Read the dataset
df = pd.read_csv('preprocessedbaru_skincare.csv')  # Assuming the dataset is available

# For simplicity, I am creating a simulated 'Liked' column as user feedback (0 = not liked, 1 = liked)
# You should replace this with actual user feedback data if available
df['Liked'] = np.random.choice([0, 1], size=len(df))

# Create a feature column for ingredients (you can modify this for other features like description)
df['features'] = df['Ingredients']

# Create a TF-IDF vectorizer and fit on the 'features' column (Ingredients or any other text-based feature)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['features'])

# Compute cosine similarity between all products
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create a series to map product names to their indices
indices = pd.Series(df.index, index=df['Name']).drop_duplicates()

# Function to recommend products based on input ingredients and other filters
def recommend_cosmetics(skin_type, label_filter, rank_filter, brand_filter, price_range, ingredient_input=None, num_recommendations=10):
    # Filter products based on skin type and other filters first
    recommended_products = df[df[skin_type] == 1]
    
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
        tfidf_matrix = vectorizer.fit_transform(df['Ingredients'])
        input_vec = vectorizer.transform([ingredient_input])
        cosine_similarities = cosine_similarity(input_vec, tfidf_matrix).flatten()
        recommended_indices = cosine_similarities.argsort()[-num_recommendations:][::-1]
        ingredient_recommendations = df.iloc[recommended_indices]
        recommended_products = recommended_products[recommended_products.index.isin(ingredient_recommendations.index)]
    
    return recommended_products.sort_values(by=['Rank']).head(num_recommendations)

# Evaluation Metrics
def evaluate_recommendations(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return accuracy, precision, recall, f1

def main():
    st.set_page_config(page_title="Skincare Products Recommendation System", layout="wide")

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
    st.markdown("---")  # Separator line
    st.subheader("🔍 Search Filters")
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        skin_type = st.selectbox(
            'Select your skin type:',
            ('Combination', 'Dry', 'Normal', 'Oily', 'Sensitive')
        )

    unique_labels = df['Label'].unique().tolist()
    unique_labels.insert(0, 'All')

    with col2:
        label_filter = st.selectbox(
            'Filter by label (optional):',
            unique_labels
        )

    with col1:
        rank_filter = st.slider(
            'Select rank range:',
            min_value=int(df['Rank'].min()),
            max_value=int(df['Rank'].max()),
            value=(int(df['Rank'].min()), int(df['Rank'].max()))
        )

    unique_brands = df['Brand'].unique().tolist()
    unique_brands.insert(0, 'All')

    with col2:
        brand_filter = st.selectbox(
            'Filter by brand (optional):',
            unique_brands
        )

    with col3:
        price_range = st.slider(
            'Select price range:',
            min_value=float(df['Price'].min()),
            max_value=float(df['Price'].max()),
            value=(float(df['Price'].min()), float(df['Price'].max()))
        )

    st.markdown("---")  # Separator line

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
                top_recommended_products[['Label', 'Brand', 'Name', 'Ingredients', 'Rank']],
                height=400
            )

            # Simulate user feedback (1 = liked, 0 = not liked)
            y_true = top_recommended_products['Liked'].values
            y_pred = np.random.choice([0, 1], size=len(y_true))  # Randomly generate predictions

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
