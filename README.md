Book Recommendation System

A personalized book recommendation system built using User-Based Collaborative Filtering, K-Nearest Neighbors (KNN), and Cosine Similarity.

The application recommends books based on the preferences of users with similar reading and rating patterns.

Live Demo

[https://book-recommendation-system-ug9n2eckgdssqcgg9dui5i.streamlit.app/](url)

How It Works
Users who have rated at least 30 books are selected.
Books with at least 50 ratings are selected.
A user-book sparse matrix is created.
Cosine similarity and KNN are used to find similar users.
Books liked by similar users are given recommendation scores.
Books already rated by the selected user are removed.
The top recommendations are displayed in the Streamlit application.

Why a Sparse Matrix?
The dataset contains thousands of users and books, but each user has rated only a small number of books.
A normal dense user-book matrix required too much memory. Therefore, a SciPy CSR sparse matrix is used to store only the available ratings, significantly reducing memory usage.

Model Evaluation
The model was evaluated on a random sample of 500 users.
For each user, 80% of the ratings were used for training and 20% for testing. Books rated 4 or higher were considered relevant.

Metric	Result
**Precision@10	0.2336
Recall@10	0.1571
Hit Rate@10	0.8580**

A Hit Rate@10 of 0.858 means that 85.8% of evaluated users received at least one relevant book among their top-10 recommendations.

Technologies
Python
Pandas
NumPy
SciPy
Scikit-learn
KNN
Collaborative Filtering
Streamlit
Git & GitHub

Files
app.py — Streamlit application and recommendation system
evaluate.py — Model evaluation
books.csv — Book information
ratings.csv — User ratings
requirements.txt — Required Python libraries

Run Locally
Install the required libraries:
pip install -r requirements.txt

Run the application:
streamlit run app.py

Run the evaluation:
python evaluate.py

Author
Dhashvinth Bashkar
