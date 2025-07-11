import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
import random
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from collections import defaultdict

class HybridRecommender:
    def __init__(self):
        self.products_df = None
        self.ratings_df = None
        self.user_item_matrix = None
        self.content_similarity_matrix = None
        self.svd_model = None
        self.model_path = 'models'
        self.placeholder_images = [
            "https://placehold.co/300x300/EEE/31343C?text=Skincare+Product",
            "https://placehold.co/300x300/FFEEEE/31343C?text=Skincare+Product",
            "https://placehold.co/300x300/EEFFEE/31343C?text=Skincare+Product",
            "https://placehold.co/300x300/EEEEFF/31343C?text=Skincare+Product",
            "https://placehold.co/300x300/FFFFEE/31343C?text=Skincare+Product",
            "https://placehold.co/300x300/EEFFFF/31343C?text=Skincare+Product",
            "https://placehold.co/300x300/FFEEFF/31343C?text=Skincare+Product"
        ]
        
        # Create models directory if it doesn't exist
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
    
    def load_data(self):
        """Load product and ratings data"""
        # Load product data
        self.products_df = pd.read_csv('data/product_info.csv')
        
        # Select only relevant columns and rename them
        self.products_df = self.products_df[['product_id', 'product_name', 'brand_name', 'primary_category', 'ingredients', 'price_usd']]
        self.products_df = self.products_df.rename(columns={
            'product_name': 'name',
            'brand_name': 'brand',
            'primary_category': 'category'
        })
        
        # Add placeholder images
        self.products_df['image_url'] = self.products_df.apply(lambda x: random.choice(self.placeholder_images), axis=1)
        
        # Extract skin type from ingredients or set to 'All' if not available
        self.products_df['skin_type'] = 'All'
        
        # Clean up data
        self.products_df = self.products_df.dropna(subset=['name', 'brand', 'category'])
        self.products_df = self.products_df.head(1000)  # Limit to 1000 products for performance
        
        # Load ratings data
        # We'll use a sample of reviews for demonstration
        self.ratings_df = pd.read_csv('data/reviews_0-250.csv', low_memory=False)
        
        # Select only relevant columns and rename them
        self.ratings_df = self.ratings_df[['author_id', 'product_id', 'rating']]
        self.ratings_df = self.ratings_df.rename(columns={'author_id': 'user_id'})
        
        # Clean up data
        self.ratings_df = self.ratings_df.dropna()
        self.ratings_df = self.ratings_df[self.ratings_df['product_id'].isin(self.products_df['product_id'])]
        
        return self.products_df, self.ratings_df
    
    def preprocess_data(self):
        """Preprocess data for recommendation models"""
        # Create a combined features column for content-based filtering
        # Convert ingredients to string if it's not already
        self.products_df['ingredients'] = self.products_df['ingredients'].astype(str)
        
        self.products_df['combined_features'] = (
            self.products_df['brand'] + ' ' + 
            self.products_df['category'] + ' ' + 
            self.products_df['ingredients'] + ' ' + 
            self.products_df['skin_type']
        )
        
        # Create user-item matrix for collaborative filtering
        # Ensure user_id and product_id are the right types
        self.ratings_df['user_id'] = self.ratings_df['user_id'].astype(str)
        self.ratings_df['product_id'] = self.ratings_df['product_id'].astype(str)
        
        # Create the pivot table
        try:
            self.user_item_matrix = self.ratings_df.pivot(
                index='user_id', 
                columns='product_id', 
                values='rating'
            ).fillna(0)
        except ValueError:
            # If there are duplicate entries, take the mean rating
            user_item_df = self.ratings_df.groupby(['user_id', 'product_id'])['rating'].mean().reset_index()
            self.user_item_matrix = user_item_df.pivot(
                index='user_id',
                columns='product_id',
                values='rating'
            ).fillna(0)
        
        return self.products_df, self.user_item_matrix
    
    def build_content_based_model(self):
        """Build content-based filtering model"""
        # Create TF-IDF vectorizer
        tfidf = TfidfVectorizer(stop_words='english')
        
        # Create TF-IDF matrix
        tfidf_matrix = tfidf.fit_transform(self.products_df['combined_features'])
        
        # Calculate cosine similarity
        self.content_similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # Save the model
        joblib.dump(self.content_similarity_matrix, f'{self.model_path}/content_similarity_matrix.joblib')
        joblib.dump(tfidf, f'{self.model_path}/tfidf_vectorizer.joblib')
        
        return self.content_similarity_matrix
    
    def build_matrix_factorization_model(self):
        """Build matrix factorization model using Surprise SVD"""
        # Prepare data for Surprise
        reader = Reader(rating_scale=(1, 5))
        
        # Create the dataset
        data = Dataset.load_from_df(self.ratings_df[['user_id', 'product_id', 'rating']], reader)
        
        # Split the data into train and test sets
        trainset = data.build_full_trainset()
        
        # Define the SVD algorithm
        self.svd_model = SVD(n_factors=20, n_epochs=20, lr_all=0.005, reg_all=0.02)
        
        # Train the model
        self.svd_model.fit(trainset)
        
        # Save the model
        joblib.dump(self.svd_model, f'{self.model_path}/surprise_svd_model.joblib')
        
        return self.svd_model
    
    def get_content_based_recommendations(self, product_id, top_n=10):
        """Get content-based recommendations based on a product"""
        # Find the index of the product
        product_indices = self.products_df[self.products_df['product_id'] == product_id].index
        if len(product_indices) == 0:
            return self.products_df.sample(top_n)
            
        idx = product_indices[0]
        
        # Get similarity scores
        sim_scores = list(enumerate(self.content_similarity_matrix[idx]))
        
        # Sort products based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N similar products (excluding the product itself)
        sim_scores = sim_scores[1:top_n+1]
        
        # Get product indices
        product_indices = [i[0] for i in sim_scores]
        
        # Return recommended products
        return self.products_df.iloc[product_indices]
    
    def get_matrix_factorization_recommendations(self, user_id, top_n=10):
        """Get matrix factorization recommendations for a user using Surprise"""
        # Get all products
        all_products = self.products_df['product_id'].unique()
        
        # Get products the user has already rated
        user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]
        already_rated_products = user_ratings['product_id'].tolist()
        
        # Find products the user hasn't rated
        products_to_predict = [prod for prod in all_products if prod not in already_rated_products]
        
        # Predict ratings for all unrated products
        predictions = []
        for product_id in products_to_predict:
            try:
                predicted_rating = self.svd_model.predict(user_id, product_id).est
                predictions.append((product_id, predicted_rating))
            except:
                # Skip if prediction fails
                continue
        
        # Sort predictions by estimated rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Get top N product IDs
        top_product_ids = [pred[0] for pred in predictions[:top_n]]
        
        # Get product details
        recommended_products = self.products_df[self.products_df['product_id'].isin(top_product_ids)].copy()
        
        # Add predicted ratings
        pred_ratings = {pred[0]: pred[1] for pred in predictions[:top_n]}
        recommended_products['predicted_rating'] = recommended_products['product_id'].map(pred_ratings)
        
        # Sort by predicted rating
        recommended_products = recommended_products.sort_values(by='predicted_rating', ascending=False)
        
        return recommended_products
    
    def get_popular_products(self, top_n=10):
        """Get popular products based on average rating and number of reviews"""
        # Group by product_id and calculate mean rating and count
        product_ratings = self.ratings_df.groupby('product_id').agg({
            'rating': ['mean', 'count']
        })
        
        # Flatten the column names
        product_ratings.columns = ['mean_rating', 'review_count']
        product_ratings = product_ratings.reset_index()
        
        # Filter products with at least 2 reviews
        popular_products = product_ratings[product_ratings['review_count'] > 2]
        
        # Sort by mean rating
        popular_products = popular_products.sort_values(by='mean_rating', ascending=False)
        
        # Get top N products
        top_products = popular_products.head(top_n)
        
        # Get product details
        recommended_products = self.products_df[self.products_df['product_id'].isin(top_products['product_id'])]
        
        # Merge with ratings data
        recommended_products = recommended_products.merge(top_products, on='product_id')
        
        # Sort by mean rating
        recommended_products = recommended_products.sort_values(by='mean_rating', ascending=False)
        
        return recommended_products
    
    def get_hybrid_recommendations(self, user_id=None, product_id=None, top_n=10):
        """Get hybrid recommendations based on user history or product similarity"""
        if user_id is not None:
            # Check if user exists in ratings data
            if user_id in self.ratings_df['user_id'].values:
                # For returning users, use matrix factorization
                return self.get_matrix_factorization_recommendations(user_id, top_n)
            else:
                # If user not found, return popular products
                return self.get_popular_products(top_n)
        elif product_id is not None:
            # For new users or content-based recommendations, use content-based filtering
            return self.get_content_based_recommendations(product_id, top_n)
        else:
            # If no user_id or product_id is provided, return popular products
            return self.get_popular_products(top_n)
    
    def train_models(self):
        """Train both recommendation models"""
        self.load_data()
        self.preprocess_data()
        self.build_content_based_model()
        self.build_matrix_factorization_model()
        
    def load_models(self):
        """Load pre-trained models"""
        self.load_data()
        self.preprocess_data()
        
        # Load content-based model
        self.content_similarity_matrix = joblib.load(f'{self.model_path}/content_similarity_matrix.joblib')
        
        # Load matrix factorization model
        self.svd_model = joblib.load(f'{self.model_path}/surprise_svd_model.joblib') 