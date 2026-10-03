import pandas as pd
import numpy as np
from scipy import stats

# 1. Create the Pandas Series
scores = pd.Series([72, 65, 88, np.nan, 54, 91, 76, np.nan, 83, 69])

# 2. Get basic descriptive statistics
print("\nDescriptive Statistics:")
print(scores.describe())

# 3. Calculate Skewness and Kurtosis (SciPy requires dropping NaNs first)
clean_scores = scores.dropna()
print("Skewness:", stats.skew(clean_scores))
print("Kurtosis:", stats.kurtosis(clean_scores))

# 4. Count missing values
missing_count = scores.isnull().sum()
print("Missing values:", missing_count)

# 5. Fill missing scores with the mean
filled_scores = scores.fillna(scores.mean())
print("Filled scores:\n", filled_scores)