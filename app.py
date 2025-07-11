import streamlit as st
import pandas as pd
import numpy as np
import os
from recommender import HybridRecommender
import time

# Page configuration
st.set_page_config(
    page_title="Skincare Product Recommender",
    page_icon="✨",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'is_returning_user' not in st.session_state:
    st.session_state.is_returning_user = False
if 'selected_product' not in st.session_state:
    st.session_state.selected_product = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None

# Function to initialize or load recommender system
@st.cache_resource
def get_recommender():
    recommender = HybridRecommender()
    
    # Check if models exist, if not, train them
    if not os.path.exists('models/content_similarity_matrix.joblib') or \
       not os.path.exists('models/surprise_svd_model.joblib'):
        with st.spinner('Training recommendation models...'):
            recommender.train_models()
    else:
        with st.spinner('Loading recommendation models...'):
            recommender.load_models()
            
    return recommender

# Function to display product cards
def display_product_cards(products_df, num_cols=5):
    if products_df is None or len(products_df) == 0:
        st.warning("No products to display.")
        return
    
    # Display products in a grid layout
    cols = st.columns(num_cols)
    
    for i, (_, product) in enumerate(products_df.iterrows()):
        if i >= 10:  # Only display 10 products at a time
            break
            
        col_idx = i % num_cols
        with cols[col_idx]:
            st.image(product['image_url'], width=150)
            st.markdown(f"**{product['name']}**")
            st.write(f"Brand: {product['brand']}")
            st.write(f"Category: {product['category']}")
            
            # Format price with two decimal places if it exists
            if 'price_usd' in product and pd.notna(product['price_usd']):
                st.write(f"Price: ${product['price_usd']:.2f}")
            
            # Show rating if available
            if 'mean_rating' in product and pd.notna(product['mean_rating']):
                st.write(f"Rating: {product['mean_rating']:.2f} ⭐")
            elif 'predicted_rating' in product and pd.notna(product['predicted_rating']):
                st.write(f"Predicted Rating: {product['predicted_rating']:.2f} ⭐")
            
            # Add a button to select this product for recommendations
            if st.button(f"Similar Products", key=f"btn_{product['product_id']}"):
                st.session_state.selected_product = product['product_id']
                st.rerun()

# Main function
def main():
    # Get recommender system
    recommender = get_recommender()
    
    # Header
    st.title("✨ Skincare Product Recommender")
    st.caption("Powered by Surprise SVD Matrix Factorization")
    
    # Sidebar for user selection
    with st.sidebar:
        st.header("User Settings")
        
        # User type selection
        user_type = st.radio(
            "Select User Type:",
            ["New User", "Returning User"]
        )
        
        if user_type == "Returning User":
            # Get a sample of unique user IDs from the ratings data
            user_ids = recommender.ratings_df['user_id'].astype(str).unique().tolist()
            if len(user_ids) > 100:
                user_ids = user_ids[:100]  # Limit to 100 users for better performance
            
            # User ID selection
            selected_user_id = st.selectbox(
                "Select User ID:",
                user_ids
            )
            
            if st.button("Load User Recommendations"):
                st.session_state.user_id = selected_user_id
                st.session_state.is_returning_user = True
                st.session_state.selected_product = None
                
                # Get recommendations for the user
                with st.spinner("Getting personalized recommendations..."):
                    st.session_state.recommendations = recommender.get_hybrid_recommendations(
                        user_id=selected_user_id,
                        top_n=10
                    )
                st.success(f"Loaded recommendations for User {selected_user_id}")
                st.rerun()
        else:
            st.session_state.is_returning_user = False
            
            # Product category filter for new users
            categories = recommender.products_df['category'].unique().tolist()
            selected_category = st.selectbox(
                "Filter by Category:",
                ["All"] + categories
            )
            
            # Brand filter for new users
            brands = recommender.products_df['brand'].unique().tolist()
            if len(brands) > 20:
                # If there are too many brands, select the most common ones
                brand_counts = recommender.products_df['brand'].value_counts()
                top_brands = brand_counts.head(20).index.tolist()
                brands = ["All"] + top_brands
            else:
                brands = ["All"] + brands
                
            selected_brand = st.selectbox(
                "Filter by Brand:",
                brands
            )
            
            # Price range filter
            min_price = float(recommender.products_df['price_usd'].min())
            max_price = float(recommender.products_df['price_usd'].max())
            
            price_range = st.slider(
                "Price Range (USD):",
                min_value=min_price,
                max_value=max_price,
                value=(min_price, max_price)
            )
            
            if st.button("Browse Products"):
                # Filter products based on selected filters
                filtered_products = recommender.products_df.copy()
                
                if selected_category != "All":
                    filtered_products = filtered_products[filtered_products['category'] == selected_category]
                    
                if selected_brand != "All":
                    filtered_products = filtered_products[filtered_products['brand'] == selected_brand]
                
                # Filter by price range
                filtered_products = filtered_products[
                    (filtered_products['price_usd'] >= price_range[0]) &
                    (filtered_products['price_usd'] <= price_range[1])
                ]
                
                st.session_state.recommendations = filtered_products.head(10)
                st.rerun()
    
    # Main content area
    if st.session_state.is_returning_user and st.session_state.user_id is not None:
        st.header(f"Personalized Recommendations for User {st.session_state.user_id}")
        st.write("Based on your previous ratings and similar users' preferences:")
        
        if st.session_state.recommendations is not None:
            display_product_cards(st.session_state.recommendations)
    
    elif st.session_state.selected_product is not None:
        # Get content-based recommendations for the selected product
        product_info = recommender.products_df[recommender.products_df['product_id'] == st.session_state.selected_product]
        
        if len(product_info) > 0:
            product_info = product_info.iloc[0]
            st.header(f"Products Similar to {product_info['name']}")
            
            with st.spinner("Finding similar products..."):
                similar_products = recommender.get_content_based_recommendations(
                    product_id=st.session_state.selected_product,
                    top_n=10
                )
                
            display_product_cards(similar_products)
        else:
            st.error("Product not found. Please select another product.")
            st.session_state.selected_product = None
        
        if st.button("Back to Browse"):
            st.session_state.selected_product = None
            st.rerun()
    
    else:
        st.header("Discover Skincare Products")
        
        if st.session_state.recommendations is not None:
            display_product_cards(st.session_state.recommendations)
        else:
            # Show popular products by default
            with st.spinner("Finding popular products..."):
                popular_products = recommender.get_hybrid_recommendations(top_n=10)
            display_product_cards(popular_products)
    
    # Footer
    st.markdown("---")
    st.caption("Skincare Product Recommender | Powered by Hybrid Recommendation System with Surprise SVD")

if __name__ == "__main__":
    main() 