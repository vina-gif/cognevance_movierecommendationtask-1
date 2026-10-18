
# Movie Recommendation System

A simple AI-based movie recommendation system built with Python. It suggests movies to users based on ratings from similar users, using **user-based collaborative filtering** and **cosine similarity**.

## Dataset

MovieLens `ml-latest-small` (GroupLens): 100,836 ratings, 610 users, 9,742 movies.

## Project Workflow

1. Load the ratings and movies datasets
2. Clean and preprocess the data (missing values, duplicates, merge)
3. Analyze ratings and genres
4. Build a user-movie matrix and compute cosine similarity
5. Recommend top 5 movies for different users
6. Create visualizations

## Files

- `movie_recommender.py` : complete source code
- `ratings.csv`, `movies.csv` : dataset
- `output.txt` : program output with recommendations
- `*.png` : charts (top 10 movies, genre count, rating distribution, ratings trend)

## How to Run

```
pip install pandas numpy matplotlib seaborn scikit-learn
python movie_recommender.py
```

## Sample Output (User 200)

| Movie | Predicted Rating |
|---|---|
| Fantastic Mr. Fox (2009) | 4.84 |
| Last of the Mohicans, The (1992) | 4.77 |
| Unforgiven (1992) | 4.68 |
| Mr. Smith Goes to Washington (1939) | 4.66 |
| Gladiator (1992) | 4.66 |

## Tech Stack

Python, Pandas, NumPy, Matplotlib, Scikit-learn
