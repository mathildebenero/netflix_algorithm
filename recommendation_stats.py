import pandas as pd
import numpy as np

def load_ratings(filepath: str) -> pd.DataFrame:
    """
    Load the ratings matrix from a CSV file.
    
    Args:
        filepath (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: DataFrame containing the ratings matrix.
    """
    ratings_df = pd.read_csv(filepath)
    return ratings_df

def fill_missing_ratings_with_movie_avg(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace missing values in the ratings matrix with the average rating of the movie.
    
    Args:
        ratings_df (pd.DataFrame): DataFrame containing the ratings matrix.
        
    Returns:
        pd.DataFrame: DataFrame with missing ratings replaced.
    """
    # Compute the average rating for each movie (column-wise mean, excluding NaN)
    movie_means = ratings_df.iloc[:, 2:].mean(axis=0, skipna=True)
    
    # Fill missing values with the corresponding movie's average rating
    filled_df = ratings_df.iloc[:, 2:].apply(lambda col: col.fillna(movie_means[col.name]))
    
    # Combine with the original metadata columns (if any)
    filled_df = pd.concat([ratings_df.iloc[:, :2], filled_df], axis=1)
    return filled_df

def calculate_stats(ratings_df: pd.DataFrame) -> dict:
    """
    Calculate basic statistics for the ratings matrix.
    
    Args:
        ratings_df (pd.DataFrame): DataFrame containing the ratings matrix.
        
    Returns:
        dict: Dictionary with calculated statistics.
    """
    ratings_only = ratings_df.iloc[:, 2:]  # Focus only on rating columns

    # Total number of ratings (non-missing values)
    total_ratings = ratings_only.count().sum()
    
    # Overall average rating
    overall_avg_rating = ratings_only.stack().mean()
    
    # User statistics: count of ratings and their average
    user_counts = ratings_only.count(axis=1)
    user_means = ratings_only.mean(axis=1)
    
    # Movie statistics: count of ratings and their average
    movie_counts = ratings_only.count(axis=0)
    movie_means = ratings_only.mean(axis=0)
    
    return {
        "total_ratings": total_ratings,
        "overall_avg_rating": overall_avg_rating,
        "user_counts": user_counts.to_dict(),
        "user_means": user_means.to_dict(),
        "movie_counts": movie_counts.to_dict(),
        "movie_means": movie_means.to_dict(),
    }
