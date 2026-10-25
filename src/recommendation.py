import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("data/games.csv")


# -----------------------------
# 2. Preprocessing
# -----------------------------

# Handle missing summaries
df["Summary"] = df["Summary"].fillna("")

# Clean genre text
df["Genres_Clean"] = (
    df["Genres"]
    .str.replace("[", "", regex=False)
    .str.replace("]", "", regex=False)
    .str.replace("'", "", regex=False)
)

# Combine genres and summary
df["Content"] = df["Genres_Clean"] + " " + df["Summary"]

# Remove duplicate game titles
df = df.drop_duplicates(
    subset="Title",
    keep="first"
).reset_index(drop=True)


# -----------------------------
# 3. TF-IDF
# -----------------------------
tfidf = TfidfVectorizer(stop_words="english")

tfidf_matrix = tfidf.fit_transform(df["Content"])


# -----------------------------
# 4. Cosine similarity
# -----------------------------
similarity_matrix = cosine_similarity(tfidf_matrix)


# -----------------------------
# 5. Recommendation function
# -----------------------------
def recommend(game_title, number_of_recommendations=5):

    matches = df.index[
        df["Title"].str.lower() == game_title.lower()
    ].tolist()

    if not matches:
        print(f"Game '{game_title}' was not found.")
        return

    game_index = matches[0]

    similarity_scores = list(
        enumerate(similarity_matrix[game_index])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    print(f"\nRecommendations for: {df.iloc[game_index]['Title']}\n")

    count = 0

    for index, score in similarity_scores:

        # Skip the selected game itself
        if index == game_index:
            continue

        print(
            f"{count + 1}. "
            f"{df.iloc[index]['Title']} "
            f"-> similarity: {score:.3f}"
        )

        count += 1

        if count == number_of_recommendations:
            break


# -----------------------------
# 6. Run the recommender
# -----------------------------
if __name__ == "__main__":

    print("GameVault ML Recommendation System")
    print(f"Dataset: {len(df)} unique games")

    game_title = input("\nEnter a game title: ")

    recommend(game_title)