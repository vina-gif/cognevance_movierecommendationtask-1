import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Setup: file paths ----------
BASE = os.path.dirname(os.path.abspath(__file__))

def find_file(name):
    for folder in [BASE, os.path.join(BASE, "Data")]:
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"{name} nahi mila.")

CHARTS = os.path.join(BASE, "Charts")
os.makedirs(CHARTS, exist_ok=True)

# ---------- TASK 1: Dataset load ----------
ratings = pd.read_csv(find_file("ratings.csv"))
movies = pd.read_csv(find_file("movies.csv"))

print("===== DATASET INFO =====")
print("Total ratings:", len(ratings))
print("Total users:", ratings["userId"].nunique())
print("Total movies:", movies["movieId"].nunique())

# ---------- TASK 2: Cleaning & preprocessing ----------
print("\n===== CLEANING =====")
print("Missing values (ratings):\n", ratings.isnull().sum())
print("Missing values (movies):\n", movies.isnull().sum())

ratings = ratings.drop_duplicates().dropna()
movies = movies.drop_duplicates().dropna()

df = ratings.merge(movies, on="movieId")
df["date"] = pd.to_datetime(df["timestamp"], unit="s")
df["year"] = df["date"].dt.year
print("Merged data shape:", df.shape)

# ---------- TASK 3: Analysis ----------
print("\n===== ANALYSIS =====")
movie_stats = df.groupby("title").agg(
    avg_rating=("rating", "mean"),
    num_ratings=("rating", "count")
)

print("\nTop 10 most rated movies:")
print(movie_stats.sort_values("num_ratings", ascending=False).head(10))

print("\nTop 10 highest rated (minimum 100 ratings):")
print(movie_stats[movie_stats["num_ratings"] >= 100]
      .sort_values("avg_rating", ascending=False).head(10))

genres_df = movies.copy()
genres_df["genres"] = genres_df["genres"].str.split("|")
genres_df = genres_df.explode("genres")
genres_df = genres_df[genres_df["genres"] != "(no genres listed)"]
genre_count = genres_df["genres"].value_counts()
print("\nMovies per genre:\n", genre_count)

# ---------- TASK 4: Collaborative Filtering (User-based) ----------
user_movie = df.pivot_table(index="userId", columns="movieId",
                            values="rating").fillna(0)

sim_matrix = cosine_similarity(user_movie)
sim_df = pd.DataFrame(sim_matrix, index=user_movie.index, columns=user_movie.index)

movie_titles = movies.set_index("movieId")["title"]

def recommend(user_id, n=5, k=20, min_neighbors=3):
    if user_id not in user_movie.index:
        return "User not found"

    similar_users = sim_df[user_id].drop(user_id).nlargest(k)
    neighbors = user_movie.loc[similar_users.index]

    weighted_sum = neighbors.mul(similar_users, axis=0).sum()
    sim_sum = (neighbors > 0).mul(similar_users, axis=0).sum()
    num_neighbors = (neighbors > 0).sum()

    scores = weighted_sum / sim_sum
    scores = scores[num_neighbors >= min_neighbors]

    seen = user_movie.loc[user_id]
    seen = seen[seen > 0].index
    scores = scores.drop(labels=seen, errors="ignore")

    top = scores.nlargest(n)
    return pd.DataFrame({
        "title": [movie_titles[m] for m in top.index],
        "predicted_rating": top.round(2).values
    })

# ---------- TASK 5: Recommendations ----------
print("\n===== RECOMMENDATIONS =====")
for uid in [1, 50, 100, 200]:
    print(f"\nTop 5 recommendations for User {uid}:")
    print(recommend(uid, n=5).to_string(index=False))

# ---------- TASK 6: Visualizations ----------
top10 = movie_stats.sort_values("num_ratings", ascending=True).tail(10)
plt.figure(figsize=(10, 6))
plt.barh(top10.index, top10["num_ratings"], color="steelblue")
plt.title("Top 10 Most Rated Movies")
plt.xlabel("Number of Ratings")
plt.tight_layout()
plt.savefig(os.path.join(CHARTS, "top10_movies.png"))
plt.close()

plt.figure(figsize=(8, 5))
df["rating"].plot(kind="hist", bins=10, color="orange", edgecolor="black")
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(CHARTS, "rating_distribution.png"))
plt.close()

plt.figure(figsize=(10, 6))
genre_count.sort_values().plot(kind="barh", color="seagreen")
plt.title("Number of Movies per Genre")
plt.xlabel("Number of Movies")
plt.tight_layout()
plt.savefig(os.path.join(CHARTS, "genre_count.png"))
plt.close()

plt.figure(figsize=(10, 5))
df.groupby("year").size().plot(kind="line", marker="o", color="crimson")
plt.title("Number of Ratings per Year")
plt.xlabel("Year")
plt.ylabel("Ratings")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS, "ratings_trend.png"))
plt.close()

print("\nDone! 4 charts 'Charts' folder mein save ho gaye.")
