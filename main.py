import pandas as pd
import numpy as np
from recommendation_stats import (
    load_ratings,
    fill_missing_ratings_with_movie_avg,
    calculate_stats
)
from svd_decomposition import (
    prepare_matrix,
    compute_svd,
    get_recommendations
)

if __name__ == "__main__":
    # Path to your uploaded ratings file
    filepath = "./matrix/CourseMovieGradings-2025.xlsx.csv"
    
    # Step 1: Load the ratings matrix
    ratings_matrix = load_ratings(filepath)
    
    # Step 2: Fill missing ratings using movie averages
    ratings_filled = fill_missing_ratings_with_movie_avg(ratings_matrix)
    
    # Step 3: Calculate statistics
    stats = calculate_stats(ratings_filled)
    
    # Step 4: Print the statistics
    print("User Stats (Counts and Means):")
    print(stats["user_counts"])
    print(stats["user_means"])
    print("\nMovie Stats (Counts and Means):")
    print(stats["movie_counts"])
    print(stats["movie_means"])
    print(f"\nTotal Ratings: {stats['total_ratings']}")
    print(f"Overall Average Rating: {stats['overall_avg_rating']}")
    
    # Step 5: Prepare the matrix for SVD
    prepared_matrix = prepare_matrix(ratings_matrix.iloc[:, 2:])  # Exclude metadata columns
    
    # Step 6: Compute SVD with k=3
    k = 3
    U, Sigma, VT = compute_svd(prepared_matrix, k)
    print("\nSVD Outputs:")
    print("U (Truncated):", U)
    print("Σ (Truncated):", Sigma)
    print("V^T (Truncated):", VT)
    
    # Step 7: Generate recommendations for a specific student
    student_name = "Mathilde"
    recommendations = get_recommendations(
        student_name, ratings_matrix, U, Sigma, VT, n_recommendations=5
    )
    print(f"\nRecommendations for {student_name}:")
    print(recommendations)
