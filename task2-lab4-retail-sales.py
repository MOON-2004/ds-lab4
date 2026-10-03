import pandas as pd
import numpy as np
from scipy.stats import zscore

# 1. Create the DataFrame
data = {
    "Day": range(1, 11),
    "Revenue": [45, 52, np.nan, 48, 390, 55, np.nan, 50, 47, 53]
}
df = pd.DataFrame(data)

# 2. Fill missing values with median
print("Missing values in Revenue:", df["Revenue"].isnull().sum())
df["Revenue"] = df["Revenue"].fillna(df["Revenue"].median())

# 3. Min-Max Normalization and Z-score Standardization
rev = df["Revenue"]
df["Revenue_MinMax"] = (rev - rev.min()) / (rev.max() - rev.min())
df["Revenue_Zscore"] = zscore(rev)

# 4. Outlier Detection using IQR method
Q1 = rev.quantile(0.25)
Q3 = rev.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 5. Flag outliers
df["Outlier"] = (rev < lower_bound) | (rev > upper_bound)
outlier_days = df.loc[df["Outlier"], "Day"].tolist()

print("IQR outliers detected on day(s):", outlier_days)
print(df)