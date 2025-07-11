# Skincare Product Recommender System

A hybrid recommender system for skincare products that combines content-based filtering and matrix factorization using the Surprise library to provide personalized recommendations.

## Features

- **Content-Based Filtering**: Recommends products similar to ones you like based on product attributes (ingredients, brand, category)
- **Matrix Factorization**: Uses Surprise SVD algorithm to recommend products based on similar users' preferences
- **Hybrid Approach**: Uses content-based filtering for new users and matrix factorization for returning users
- **User-Friendly Interface**: Streamlit web application with product images and details

## Installation

1. Clone this repository:
```
git clone <repository-url>
cd skincare-recommender
```

2. Install the required packages:
```
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```
streamlit run app.py
```

2. Open your web browser and go to `http://localhost:8501`

3. Use the app:
   - **New Users**: Browse products by category, brand, and price range, or select a product to see similar items
   - **Returning Users**: Select your user ID to get personalized recommendations based on your previous ratings

## Data

The system uses two main datasets:
- `data/product_info.csv`: Contains product information (name, brand, category, ingredients, price, etc.)
- `data/reviews_*.csv`: Contains user ratings and reviews for products

## How It Works

### For New Users
- The system uses content-based filtering to recommend products similar to ones you select
- Products are recommended based on similarity in ingredients, brand, and category

### For Returning Users
- The system uses Surprise's SVD algorithm to analyze patterns in user-product interactions
- Recommendations are based on predicted ratings for products the user hasn't rated yet

### Surprise SVD Algorithm
- The Surprise library's SVD implementation is similar to Simon Funk's SVD approach used in the Netflix Prize
- It uses stochastic gradient descent to minimize the regularized squared error on the set of known ratings
- Parameters:
  - n_factors: Number of latent factors (20)
  - n_epochs: Number of iterations of SGD (20)
  - lr_all: Learning rate for all parameters (0.005)
  - reg_all: Regularization term for all parameters (0.02)

## Project Structure

- `app.py`: Streamlit web application
- `recommender.py`: Implementation of the hybrid recommender system
- `data/`: Directory containing the datasets
- `models/`: Directory for storing trained models

## Requirements

See `requirements.txt` for the complete list of dependencies. 