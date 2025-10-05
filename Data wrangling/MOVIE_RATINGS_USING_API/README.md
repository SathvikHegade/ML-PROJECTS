# 🎬 TMDB Top-Rated Movies Data

[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## 🔹 Project Overview
This project fetches **top-rated movies** data from **TMDB (The Movie Database) API**, converts it into a **Pandas DataFrame**, and cleans the data by removing **duplicates** and handling missing values.

The final DataFrame is ready for further use in **data analysis, visualization, or machine learning projects**.

---

## 🔹 Steps to Achieve This

1. **Get TMDB API Key**  
   - Sign up at [TMDB](https://www.themoviedb.org/) and generate an API key.

2. **Install Required Libraries**  
   ```bash
   pip install pandas requests
   ```

3. **Fetch Data from TMDB API**  
   - Use `requests` to access multiple pages of top-rated movies.

4. **Convert JSON Response to DataFrame**  
   - Select columns: `id`, `title`, `overview`, `release_date`, `popularity`, `vote_average`, `vote_count`.  
   - Handle missing columns by filling with `NaN`.

5. **Handle Data**  
   - Remove duplicate rows (all columns or specific columns like `title`).  
   - Reset the index for a clean DataFrame.

---

## 🔹 Sample Code

```python
import requests
import pandas as pd

API_KEY = "YOUR_API_KEY_HERE"
df_list = []

for i in range(1, 518):
    url = f"https://api.themoviedb.org/3/movie/top_rated?api_key={API_KEY}&language=en-US&page={i}"
    response = requests.get(url)
    
    if response.status_code == 200:
        results = response.json().get('results', [])
        if results:
            temp_df = pd.DataFrame(results)
            cols = ["id","title","overview","release_date","popularity","vote_average","vote_count"]
            existing_cols = [c for c in cols if c in temp_df.columns]
            temp_df = temp_df[existing_cols]
            df_list.append(temp_df)

# Combine all pages into one DataFrame
df = pd.concat(df_list, ignore_index=True)

# Remove duplicates based on all columns
df = df.drop_duplicates(keep='first')

# Reset index
df = df.reset_index(drop=True)

print(df)
```

---

## 🔹 Libraries Used
- `pandas` – Data handling and cleaning  
- `requests` – Fetching data from API

---

## 🔹 How to Use
1. Clone this repository.  
2. Replace `API_KEY` with your TMDB API key.  
3. Run the Python script to fetch and clean the data.  
4. Access the cleaned DataFrame for further use.

---

## 🔹 License
This project is licensed under the **MIT License**.