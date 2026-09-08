import streamlit as st
import pandas as pd
import numpy as np

from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)


# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_data():

    base_dir = Path(__file__).resolve().parent

    ratings = pd.read_csv(
        base_dir / "ratings.csv"
    )

    books = pd.read_csv(
        base_dir / "books.csv"
    )

    return ratings, books


ratings, books = load_data()


# ==========================================================
# TITLE
# ==========================================================

st.title("📚 Book Recommendation System")

st.write(
    "Select a user and get personalized book recommendations "
    "based on the preferences of similar readers."
)

st.divider()


# ==========================================================
# FILTER RATINGS DATA
# ==========================================================

@st.cache_data
def filter_data(ratings):

    # ------------------------------------------------------
    # KEEP ACTIVE USERS
    # ------------------------------------------------------

    user_rating_counts = ratings[
        "user_id"
    ].value_counts()

    active_users = user_rating_counts[
        user_rating_counts >= 30
    ].index

    filtered_ratings = ratings[
        ratings["user_id"].isin(active_users)
    ].copy()


    # ------------------------------------------------------
    # KEEP POPULAR BOOKS
    # ------------------------------------------------------

    book_rating_counts = filtered_ratings[
        "book_id"
    ].value_counts()

    popular_books = book_rating_counts[
        book_rating_counts >= 50
    ].index

    filtered_ratings = filtered_ratings[
        filtered_ratings["book_id"].isin(popular_books)
    ].copy()


    return filtered_ratings


filtered_ratings = filter_data(
    ratings
)


# ==========================================================
# CHECK FILTERED DATA
# ==========================================================

if filtered_ratings.empty:

    st.error(
        "No ratings remain after filtering. "
        "Try reducing the minimum rating thresholds."
    )

    st.stop()


# ==========================================================
# CREATE USER AND BOOK MAPPINGS
# ==========================================================

@st.cache_data
def create_mappings(filtered_ratings):

    unique_users = sorted(
        filtered_ratings[
            "user_id"
        ].unique()
    )

    unique_books = sorted(
        filtered_ratings[
            "book_id"
        ].unique()
    )


    user_to_index = {
        user_id: index
        for index, user_id in enumerate(unique_users)
    }


    index_to_user = {
        index: user_id
        for user_id, index in user_to_index.items()
    }


    book_to_index = {
        book_id: index
        for index, book_id in enumerate(unique_books)
    }


    index_to_book = {
        index: book_id
        for book_id, index in book_to_index.items()
    }


    return (
        user_to_index,
        index_to_user,
        book_to_index,
        index_to_book
    )


(
    user_to_index,
    index_to_user,
    book_to_index,
    index_to_book
) = create_mappings(
    filtered_ratings
)


# ==========================================================
# CREATE SPARSE USER-BOOK MATRIX
# ==========================================================

def create_sparse_matrix(
    filtered_ratings,
    user_to_index,
    book_to_index
):

    rows = filtered_ratings[
        "user_id"
    ].map(
        user_to_index
    ).to_numpy()


    columns = filtered_ratings[
        "book_id"
    ].map(
        book_to_index
    ).to_numpy()


    values = filtered_ratings[
        "rating"
    ].astype(
        np.float32
    ).to_numpy()


    sparse_matrix = csr_matrix(
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


    return sparse_matrix


user_book_matrix = create_sparse_matrix(
    filtered_ratings,
    user_to_index,
    book_to_index
)


# ==========================================================
# TRAIN NEAREST NEIGHBOR MODEL
# ==========================================================

# IMPORTANT:
# Do NOT use @st.cache_resource here.
# The scipy sparse matrix can cause Streamlit hashing errors.

def train_model(user_book_matrix):

    number_of_users = user_book_matrix.shape[0]

    number_of_neighbors = min(
        11,
        number_of_users
    )


    model = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_neighbors=number_of_neighbors
    )


    model.fit(
        user_book_matrix
    )


    return model


knn_model = train_model(
    user_book_matrix
)


# ==========================================================
# RECOMMENDATION FUNCTION
# ==========================================================

def recommend_books(
    user_id,
    n_recommendations=10
):

    # ------------------------------------------------------
    # CHECK IF USER EXISTS
    # ------------------------------------------------------

    if user_id not in user_to_index:

        return pd.DataFrame()


    # ------------------------------------------------------
    # GET SELECTED USER INDEX
    # ------------------------------------------------------

    user_index = user_to_index[
        user_id
    ]


    # ------------------------------------------------------
    # GET USER VECTOR
    # ------------------------------------------------------

    user_vector = user_book_matrix[
        user_index
    ]


    # ------------------------------------------------------
    # FIND SIMILAR USERS
    # ------------------------------------------------------

    number_of_users = user_book_matrix.shape[0]

    number_of_neighbors = min(
        11,
        number_of_users
    )


    distances, indices = knn_model.kneighbors(
        user_vector,
        n_neighbors=number_of_neighbors
    )


    # ------------------------------------------------------
    # REMOVE SELECTED USER ITSELF
    # ------------------------------------------------------

    similar_user_indices = []
    similarity_scores = []


    for neighbor_index, distance in zip(
        indices[0],
        distances[0]
    ):

        if neighbor_index == user_index:
            continue


        similar_user_indices.append(
            neighbor_index
        )


        similarity_scores.append(
            1 - distance
        )


    # Keep maximum 10 similar users
    similar_user_indices = (
        similar_user_indices[:10]
    )

    similarity_scores = (
        similarity_scores[:10]
    )


    # ------------------------------------------------------
    # BOOKS ALREADY RATED BY SELECTED USER
    # ------------------------------------------------------

    already_rated_indices = set(
        user_vector.indices
    )


    # ------------------------------------------------------
    # CREATE RECOMMENDATION SCORES
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

            # Do not recommend books already rated
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
    # CHECK IF RECOMMENDATIONS EXIST
    # ------------------------------------------------------

    if not recommendation_scores:

        return pd.DataFrame()


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

    recommendation_list = []


    for (
        book_index,
        recommendation_score
    ) in sorted_books:

        book_id = index_to_book[
            book_index
        ]


        recommendation_list.append(
            {
                "book_id": book_id,
                "recommendation_score":
                    recommendation_score
            }
        )


    recommendation_df = pd.DataFrame(
        recommendation_list
    )


    # ------------------------------------------------------
    # MATCH BOOK ID DATA TYPES
    # ------------------------------------------------------

    try:

        recommendation_df[
            "book_id"
        ] = recommendation_df[
            "book_id"
        ].astype(
            books["book_id"].dtype
        )

    except Exception:

        pass


    # ------------------------------------------------------
    # ADD BOOK DETAILS
    # ------------------------------------------------------

    recommendation_df = recommendation_df.merge(
        books,
        on="book_id",
        how="left"
    )


    return recommendation_df


# ==========================================================
# STREAMLIT USER INTERFACE
# ==========================================================

st.subheader(
    "Choose a User"
)


available_users = sorted(
    user_to_index.keys()
)


selected_user = st.selectbox(
    "Select User ID",
    available_users
)


# ==========================================================
# NUMBER OF RECOMMENDATIONS
# ==========================================================

number_of_books = st.slider(
    "Number of Recommendations",
    min_value=5,
    max_value=20,
    value=10,
    step=1
)


# ==========================================================
# RECOMMEND BUTTON
# ==========================================================

if st.button(
    "Recommend Books",
    type="primary"
):

    with st.spinner(
        "Finding books you may like..."
    ):

        recommendations = recommend_books(
            selected_user,
            number_of_books
        )


    # ------------------------------------------------------
    # NO RECOMMENDATIONS
    # ------------------------------------------------------

    if recommendations.empty:

        st.warning(
            "No recommendations could be generated "
            "for this user."
        )


    # ------------------------------------------------------
    # DISPLAY RECOMMENDATIONS
    # ------------------------------------------------------

    else:

        st.success(
            "Recommendations generated successfully!"
        )


        st.subheader(
            f"Recommended Books for User {selected_user}"
        )


        # --------------------------------------------------
        # COLUMNS TO DISPLAY
        # --------------------------------------------------

        possible_columns = [
            "title",
            "authors",
            "average_rating",
            "original_publication_year"
        ]


        display_columns = [
            column
            for column in possible_columns
            if column in recommendations.columns
        ]


        # --------------------------------------------------
        # DISPLAY TABLE
        # --------------------------------------------------

        if display_columns:

            st.dataframe(
                recommendations[
                    display_columns
                ],
                use_container_width=True,
                hide_index=True
            )


        else:

            st.dataframe(
                recommendations,
                use_container_width=True,
                hide_index=True
            )