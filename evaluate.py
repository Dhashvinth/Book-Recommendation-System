import pandas as pd
import numpy as np

from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split


# ==========================================================
# LOAD DATA
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

ratings = pd.read_csv(
    BASE_DIR / "ratings.csv"
)

books = pd.read_csv(
    BASE_DIR / "books.csv"
)


# ==========================================================
# FILTER DATA
# ==========================================================

# Keep users who rated at least 30 books
user_rating_counts = ratings[
    "user_id"
].value_counts()

active_users = user_rating_counts[
    user_rating_counts >= 30
].index

filtered_ratings = ratings[
    ratings["user_id"].isin(active_users)
].copy()


# Keep books that received at least 50 ratings
book_rating_counts = filtered_ratings[
    "book_id"
].value_counts()

popular_books = book_rating_counts[
    book_rating_counts >= 50
].index

filtered_ratings = filtered_ratings[
    filtered_ratings["book_id"].isin(popular_books)
].copy()


print(
    f"Ratings after filtering: {len(filtered_ratings)}"
)

print(
    f"Unique users after filtering: "
    f"{filtered_ratings['user_id'].nunique()}"
)

print(
    f"Unique books after filtering: "
    f"{filtered_ratings['book_id'].nunique()}"
)


# ==========================================================
# TRAIN-TEST SPLIT BY USER
# ==========================================================

train_parts = []
test_parts = []


for user_id, user_data in filtered_ratings.groupby(
    "user_id"
):

    if len(user_data) < 5:
        continue

    train_user, test_user = train_test_split(
        user_data,
        test_size=0.20,
        random_state=42
    )

    train_parts.append(
        train_user
    )

    test_parts.append(
        test_user
    )


train_ratings = pd.concat(
    train_parts,
    ignore_index=True
)

test_ratings = pd.concat(
    test_parts,
    ignore_index=True
)


print(
    f"Training ratings: {len(train_ratings)}"
)

print(
    f"Testing ratings: {len(test_ratings)}"
)


# ==========================================================
# CREATE USER AND BOOK MAPPINGS
# ==========================================================

unique_users = sorted(
    train_ratings[
        "user_id"
    ].unique()
)

unique_books = sorted(
    train_ratings[
        "book_id"
    ].unique()
)


user_to_index = {
    user_id: index
    for index, user_id in enumerate(
        unique_users
    )
}


index_to_user = {
    index: user_id
    for user_id, index in user_to_index.items()
}


book_to_index = {
    book_id: index
    for index, book_id in enumerate(
        unique_books
    )
}


index_to_book = {
    index: book_id
    for book_id, index in book_to_index.items()
}


# ==========================================================
# CREATE SPARSE USER-BOOK MATRIX
# ==========================================================

rows = train_ratings[
    "user_id"
].map(
    user_to_index
).to_numpy()


columns = train_ratings[
    "book_id"
].map(
    book_to_index
).to_numpy()


values = train_ratings[
    "rating"
].astype(
    np.float32
).to_numpy()


user_book_matrix = csr_matrix(
    (
        values,
        (
            rows,
            columns
        )
    ),
    shape=(
        len(user_to_index),
        len(book_to_index)
    ),
    dtype=np.float32
)


print(
    f"Sparse matrix shape: "
    f"{user_book_matrix.shape}"
)


# ==========================================================
# TRAIN KNN MODEL
# ==========================================================

number_of_neighbors = min(
    11,
    user_book_matrix.shape[0]
)


knn_model = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_neighbors=number_of_neighbors
)


knn_model.fit(
    user_book_matrix
)


print(
    "KNN model trained successfully."
)


# ==========================================================
# RECOMMENDATION FUNCTION
# ==========================================================

def recommend_books(
    user_id,
    n_recommendations=10
):

    if user_id not in user_to_index:
        return []


    # ------------------------------------------------------
    # USER INDEX
    # ------------------------------------------------------

    user_index = user_to_index[
        user_id
    ]


    # ------------------------------------------------------
    # USER VECTOR
    # ------------------------------------------------------

    user_vector = user_book_matrix[
        user_index
    ]


    # ------------------------------------------------------
    # FIND NEAREST USERS
    # ------------------------------------------------------

    distances, indices = knn_model.kneighbors(
        user_vector,
        n_neighbors=number_of_neighbors
    )


    similar_user_indices = []
    similarity_scores = []


    for neighbor_index, distance in zip(
        indices[0],
        distances[0]
    ):

        # Skip the same user
        if neighbor_index == user_index:
            continue


        similar_user_indices.append(
            neighbor_index
        )


        similarity_scores.append(
            1 - distance
        )


    # Keep top 10 similar users
    similar_user_indices = (
        similar_user_indices[:10]
    )

    similarity_scores = (
        similarity_scores[:10]
    )


    # ------------------------------------------------------
    # BOOKS USER ALREADY RATED
    # ------------------------------------------------------

    already_rated_indices = set(
        user_vector.indices
    )


    # ------------------------------------------------------
    # BUILD RECOMMENDATION SCORES
    # ------------------------------------------------------

    recommendation_scores = {}


    for (
        similar_user_index,
        similarity_score
    ) in zip(
        similar_user_indices,
        similarity_scores
    ):

        similar_user_vector = (
            user_book_matrix[
                similar_user_index
            ]
        )


        rated_book_indices = (
            similar_user_vector.indices
        )


        rating_values = (
            similar_user_vector.data
        )


        for (
            book_index,
            rating
        ) in zip(
            rated_book_indices,
            rating_values
        ):

            # Do not recommend books already seen
            if book_index in already_rated_indices:
                continue


            weighted_score = (
                float(similarity_score)
                * float(rating)
            )


            recommendation_scores[
                book_index
            ] = (
                recommendation_scores.get(
                    book_index,
                    0
                )
                + weighted_score
            )


    # ------------------------------------------------------
    # NO RECOMMENDATIONS
    # ------------------------------------------------------

    if not recommendation_scores:
        return []


    # ------------------------------------------------------
    # SORT RECOMMENDATIONS
    # ------------------------------------------------------

    sorted_books = sorted(
        recommendation_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )


    sorted_books = sorted_books[
        :n_recommendations
    ]


    # ------------------------------------------------------
    # CONVERT MATRIX INDEX BACK TO BOOK ID
    # ------------------------------------------------------

    recommended_book_ids = []


    for book_index, score in sorted_books:

        book_id = index_to_book[
            book_index
        ]

        recommended_book_ids.append(
            book_id
        )


    return recommended_book_ids


# ==========================================================
# EVALUATION FUNCTION
# ==========================================================

def evaluate_user(
    user_id,
    n_recommendations=10,
    minimum_relevant_rating=4
):

    if user_id not in user_to_index:
        return None


    # ------------------------------------------------------
    # GET RECOMMENDATIONS
    # ------------------------------------------------------

    recommended_book_ids = recommend_books(
        user_id=user_id,
        n_recommendations=n_recommendations
    )


    if len(recommended_book_ids) == 0:
        return None


    # ------------------------------------------------------
    # GET USER'S TEST DATA
    # ------------------------------------------------------

    user_test_data = test_ratings[
        test_ratings["user_id"] == user_id
    ]


    # ------------------------------------------------------
    # DEFINE RELEVANT BOOKS
    # ------------------------------------------------------

    relevant_books = user_test_data[
        user_test_data["rating"]
        >= minimum_relevant_rating
    ]


    relevant_book_ids = set(
        relevant_books[
            "book_id"
        ].tolist()
    )


    # No relevant hidden books
    if len(relevant_book_ids) == 0:
        return None


    # ------------------------------------------------------
    # CONVERT RECOMMENDATIONS TO SET
    # ------------------------------------------------------

    recommended_set = set(
        recommended_book_ids
    )


    # ------------------------------------------------------
    # CALCULATE HITS
    # ------------------------------------------------------

    hits = len(
        recommended_set.intersection(
            relevant_book_ids
        )
    )


    # ------------------------------------------------------
    # PRECISION@10
    # ------------------------------------------------------

    precision = (
        hits
        / n_recommendations
    )


    # ------------------------------------------------------
    # RECALL@10
    # ------------------------------------------------------

    recall = (
        hits
        / len(relevant_book_ids)
    )


    # ------------------------------------------------------
    # HIT RATE@10
    # ------------------------------------------------------

    hit_rate = (
        1
        if hits > 0
        else 0
    )


    return {
        "user_id": user_id,
        "precision": precision,
        "recall": recall,
        "hit_rate": hit_rate,
        "hits": hits,
        "relevant_books": len(
            relevant_book_ids
        )
    }


# ==========================================================
# SELECT 500 USERS FOR EVALUATION
# ==========================================================

all_test_users = test_ratings[
    "user_id"
].unique()


# Reproducible random selection
np.random.seed(42)


sample_size = min(
    500,
    len(all_test_users)
)


sampled_users = np.random.choice(
    all_test_users,
    size=sample_size,
    replace=False
)


print("\n")
print("=" * 60)

print(
    f"Evaluating {sample_size} randomly selected users..."
)

print("=" * 60)


# ==========================================================
# EVALUATE 500 USERS
# ==========================================================

results = []


for count, user_id in enumerate(
    sampled_users,
    start=1
):

    result = evaluate_user(
        user_id=user_id,
        n_recommendations=10,
        minimum_relevant_rating=4
    )


    if result is not None:

        results.append(
            result
        )


    # Show progress every 25 users
    if count % 25 == 0:

        print(
            f"Completed {count}/{sample_size} users"
        )


# ==========================================================
# RESULTS DATAFRAME
# ==========================================================

results_df = pd.DataFrame(
    results
)


# ==========================================================
# FINAL EVALUATION RESULTS
# ==========================================================

print("\n")
print("=" * 60)
print("BOOK RECOMMENDATION SYSTEM EVALUATION")
print("=" * 60)


if results_df.empty:

    print(
        "No users could be evaluated."
    )


else:

    precision_at_10 = (
        results_df[
            "precision"
        ].mean()
    )


    recall_at_10 = (
        results_df[
            "recall"
        ].mean()
    )


    hit_rate_at_10 = (
        results_df[
            "hit_rate"
        ].mean()
    )


    print(
        f"Sampled users        : {sample_size}"
    )

    print(
        f"Users actually evaluated: "
        f"{len(results_df)}"
    )


    print(
        f"Precision@10         : "
        f"{precision_at_10:.4f}"
    )


    print(
        f"Recall@10            : "
        f"{recall_at_10:.4f}"
    )


    print(
        f"Hit Rate@10          : "
        f"{hit_rate_at_10:.4f}"
    )


    print("=" * 60)


    # ======================================================
    # SAMPLE USER RESULTS
    # ======================================================

    print(
        "\nSample user-level results:\n"
    )


    print(
        results_df.head(10)
    )


    # ======================================================
    # SAVE RESULTS
    # ======================================================

    results_df.to_csv(
        BASE_DIR / "evaluation_results.csv",
        index=False
    )


    print(
        "\nDetailed results saved as "
        "'evaluation_results.csv'."
    )