
# 🎵 Spotify Music Recommendation Application

A machine learning-based music recommendation system built using Spotify’s audio feature datasets and Spotipy (a lightweight Python library for the Spotify Web API). It recommends songs based on user input using clustering, dimensionality reduction, and cosine similarity on audio features.

---

## 📚 Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Sources](#data-sources)
- [Dependencies](#dependencies)
- [Configuration](#configuration)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributors](#contributors)
- [License](#license)

---

## 📝 Introduction

This application analyzes Spotify song data to cluster tracks based on audio features and recommend new music based on songs a user already likes. By leveraging machine learning techniques like K-Means clustering, PCA, and t-SNE, the system understands music similarities and builds an interactive recommendation system using Spotify's API.

---

## ✨ Features

- 🎼 Cluster music by genre and audio features using K-Means
- 📉 Visualize data using PCA and t-SNE
- 🔍 Analyze correlation between features and popularity
- 🎧 Recommend similar tracks using cosine similarity
- 🧠 Extract audio features using Spotify Web API
- 📊 Analyze music trends by decade and genre

---

## 💾 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/spotify-music-recommender.git
   cd spotify-music-recommender
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Spotipy**:
   ```bash
   pip install spotipy
   ```

---

## 🚀 Usage

### Run the Application

```python
python app.py
```

### Recommend Songs

Example:
```python
recommend_songs([
    {'name': 'Antidote', 'year': 2015},
    {'name': 'See You Again (feat. Charlie Puth)', 'year': 2015}
], data)
```

---

## 📂 Project Structure

```
spotify-music-recommender/
│
├── data/
│   ├── data.csv
│   ├── data_by_genres.csv
│   └── data_by_year.csv
├── visualizations/
│   └── plots and embeddings
├── app.py
├── recommendation.py
├── clustering.py
└── README.md
```

---

## 📊 Data Sources

- `data.csv`: Main Spotify dataset with ~170,000 tracks
- `data_by_genres.csv`: Aggregated audio features by genre
- `data_by_year.csv`: Aggregated audio features by year

---

## 📦 Dependencies

- Python 3.x
- pandas
- numpy
- seaborn
- matplotlib
- plotly
- scikit-learn
- yellowbrick
- spotipy
- scipy

---

## ⚙️ Configuration

1. **Set up Spotify Developer Credentials**:

Create an app at [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/).

Set the environment variables in your script:
```python
os.environ["SPOTIFY_CLIENT_ID"] = "your_client_id"
os.environ["SPOTIFY_CLIENT_SECRET"] = "your_client_secret"
```

---

## 🧪 Examples

### Grunge Music Recommendations
```python
recommend_songs([
    {'name': 'Come As You Are', 'year':1991},
    {'name': 'Smells Like Teen Spirit', 'year': 1991},
    {'name': 'Lithium', 'year': 1992},
    {'name': 'All Apologies', 'year': 1993},
], data)
```

Returns similar tracks such as:
- *Hanging By A Moment* – Lifehouse
- *Otherside* – Red Hot Chili Peppers
- *No Excuses* – Alice In Chains

---

## 🛠 Troubleshooting

- **Missing songs**: If a song isn't found in the local dataset, the app queries the Spotify API directly.
- **Rate Limiting**: If using the Spotify API too frequently, you may need to wait due to rate limits.
- **Cluster Mislabeling**: Ensure your features are correctly scaled before clustering.

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
