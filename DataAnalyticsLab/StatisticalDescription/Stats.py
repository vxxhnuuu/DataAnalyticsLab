import pandas as pd
import statistics
from scipy import stats

#Read data
data = pd.read_csv('DataAnalyticsLab\StatisticalDescription\Stats.csv')
data_column = data['Data']

#Calculate mean median and mode
mean = statistics.mean(data_column)
median = statistics.median(data_column)
mode = statistics.mode(data_column)

#Calculate modality
modes = stats.mode(data_column)
if len(modes) == 1:
    modality = "Unimodal"
if len(modes) == 2:
    modality = "Bimodal"
else:
    modality = "Multimodal"

# Calculate quartiles
q1 = statistics.quantiles(data_column , n=4)[0]
q3 = statistics.quantiles(data_column , n=4)[-1]
iqr = q3 - q1

# Identify outliers
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = [value for value in data_column if value < lower_bound or value > upper_bound]

#Display mean median mode
print(f"\nMean = {mean}")
print(f"\nMedian = {median}")
print(f"\nMode = {mode}")

#Display Modality
print(f"\nModality = {modality}")

# Display quartiles and outliers
print("Quartiles:")
print("Q1:", q1)
print("Q3:", q3)
print("IQR:", iqr)

print("\nOutliers:")
print(outliers)