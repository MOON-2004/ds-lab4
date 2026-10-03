import pandas as pd
import numpy as np
from scipy import stats

data = {
    "Hour": range(1, 16),
    "Temperature": [22, 23, np.nan, np.nan, np.nan, 25, 26, 58, 24, 23, 22, 21, 23, 24, 25]
}
df = pd.DataFrame(data)

# Save original for comparison
df["Temp_raw"] = df["Temperature"].copy()

# 1. Stats before cleaning
clean_temp = df["Temperature"].dropna()
print("Stats BEFORE cleaning:")
print("Mean:", df["Temperature"].mean(), "Std:", df["Temperature"].std())
print("Skewness:", stats.skew(clean_temp), "Kurtosis:", stats.kurtosis(clean_temp))

# 2. Interpolate missing values
df["Temperature"] = df["Temperature"].interpolate(method="linear")
print("\nAfter interpolation, missing values:", df["Temperature"].isnull().sum())

# 3. Detect Z-score outlier (>2)
temp_z = stats.zscore(df["Temperature"])
outlier_mask = np.abs(temp_z) > 2

outlier_hour = df.loc[outlier_mask, "Hour"].values[0]
outlier_val = df.loc[outlier_mask, "Temperature"].values[0]
print(f"\nZ-score(>2) outlier detected at Hour: {outlier_hour} (value: {outlier_val})")

# 4. Replace ONLY the outlier with the column median
median_temp = df["Temperature"].median()
df.loc[outlier_mask, "Temperature"] = median_temp
print("Replaced with column median:", median_temp)

# 5. Apply Z-score standardization to the cleaned data
df["Temp_clean"] = df["Temperature"]
df["Temp_Zscore"] = stats.zscore(df["Temp_clean"])

print("\nBefore/After comparison (first 8 rows):")
print(df[["Hour", "Temp_raw", "Temp_clean", "Temp_Zscore"]].head(8))

print("\nStats AFTER cleaning:")
print("Mean:", df["Temp_clean"].mean(), "Std:", df["Temp_clean"].std())