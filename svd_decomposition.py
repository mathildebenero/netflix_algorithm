import pandas as pd
import numpy as np
from scipy.linalg import svd

def prepare_matrix(ratings_df: pd.DataFrame) -> np.ndarray:
    """
    Prepare the ratings matrix for SVD by filling missing values and centering.
    
    Args:
        ratings_df (pd.DataFrame): DataFrame containing the ratings matrix.
        
    Returns:
        np.ndarray: The centered and filled ratings matrix.
    """
    # Fill missing values with the movie's average rating
    movie_means = ratings_df.mean(axis=0, skipna=True)
    filled_matrix = ratings_df.apply(lambda col: col.fillna(movie_means[col.name]))
    
    # Center the matrix by subtracting row-wise mean
    user_means = filled_matrix.mean(axis=1)
    centered_matrix = filled_matrix.subtract(user_means, axis=0)
    
    return centered_matrix.to_numpy()

def compute_svd(matrix: np.ndarray, k: int) -> tuple:
    """
    Compute the truncated SVD of the given matrix.
    
    Args:
        matrix (np.ndarray): The matrix to decompose.
        k (int): The number of singular values to retain.
        
    Returns:
        tuple: Truncated SVD components (U, Σ, V^T).
    """
    # Perform full SVD decomposition
    U, Sigma, VT = svd(matrix, full_matrices=False)
    
    # Truncate to keep only the top-k singular values
    U_truncated = U[:, :k]
    Sigma_truncated = np.diag(Sigma[:k])
    VT_truncated = VT[:k, :]
    
    return U_truncated, Sigma_truncated, VT_truncated

def get_recommendations(
    student_name: str,
    ratings_df: pd.DataFrame,
    U: np.ndarray,
    S: np.ndarray,
    VT: np.ndarray,
    n_recommendations: int = 5
) -> list:
    """
    Generate movie recommendations for a given student.
    
    Args:
        student_name (str): Name of the student.
        ratings_df (pd.DataFrame): Original ratings DataFrame.
        U (np.ndarray): Left singular matrix from SVD.
        S (np.ndarray): Singular values (diagonal matrix) from SVD.
        VT (np.ndarray): Right singular matrix from SVD.
        n_recommendations (int): Number of recommendations to return.
    
    Returns:
        list: List of recommended movie names.
    """
    # Find the column corresponding to the student
    if student_name not in ratings_df.columns:
        raise ValueError(f"Student {student_name} not found in the dataset.")
    student_col_index = ratings_df.columns.get_loc(student_name)
    
    # Reconstruct the predicted ratings for all movies
    reconstructed_matrix = U @ S @ VT
    student_ratings = reconstructed_matrix[:, student_col_index - 2]  # Adjust for metadata columns
    
    # Identify movies the student has not rated
    original_ratings = ratings_df[student_name].to_numpy()
    unrated_movies = np.where(np.isnan(original_ratings))[0]  # Indices of NaN values
    
    # Rank unrated movies by predicted ratings
    recommendations = [
        (ratings_df.iloc[i, 1], student_ratings[i])  # Use the "Movie name" column
        for i in unrated_movies
    ]
    recommendations.sort(key=lambda x: x[1], reverse=True)  # Sort by predicted rating
    
    # Return the top-n recommended movies
    return [movie for movie, _ in recommendations[:n_recommendations]]

