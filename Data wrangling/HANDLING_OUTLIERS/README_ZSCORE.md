# Z-Score Based Outlier Detection

This guide explains how to detect and remove outliers using the Z-score
method in Python with Pandas.

------------------------------------------------------------------------

## Steps

### 1. Import Required Libraries

We use **pandas** to handle the dataset.

``` python
import pandas as pd
```

------------------------------------------------------------------------

### 2. Load the Dataset

Read your dataset into a DataFrame.

``` python
df = pd.read_csv("your_dataset.csv")  # Replace with your file path
```

------------------------------------------------------------------------

### 3. Calculate Mean and Standard Deviation

We calculate mean and standard deviation of the column (here, `cgpa`).

``` python
mean = df["cgpa"].mean()
std = df["cgpa"].std()
```

------------------------------------------------------------------------

### 4. Define Upper and Lower Bounds

Use the Z-score formula: **mean ± 3 × std**.

``` python
upper_bound = mean + 3*std
lower_bound = mean - 3*std
```

------------------------------------------------------------------------

### 5. Detect Outliers

Filter rows that lie outside the bounds.

``` python
outliers = df[(df["cgpa"] < lower_bound) | (df["cgpa"] > upper_bound)]
print(outliers)
```

------------------------------------------------------------------------

### 6. Remove Outliers (Keep Only Valid Data)

``` python
df_clean = df[(df["cgpa"] >= lower_bound) & (df["cgpa"] <= upper_bound)]
```

------------------------------------------------------------------------

### 7. (Optional) Add Z-Score Column

This allows checking how far each value is from the mean.

``` python
df["zscore"] = (df["cgpa"] - mean) / std
```

------------------------------------------------------------------------

## 📊 Concept Recap

-   **Z-Score Formula:**\
    Z = (x - μ) / σ
    -   x = data point\
    -   μ = mean\
    -   σ = standard deviation
-   **Rule of Thumb (Empirical Rule):**
    -   68% of data lies within ±1σ\
    -   95% within ±2σ\
    -   99.7% within ±3σ

Values outside ±3σ are **outliers**.
