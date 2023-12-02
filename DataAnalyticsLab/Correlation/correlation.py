import pandas as pd
import numpy as np

#Read csv
data = pd.read_csv('DataAnalyticsLab\Correlation\correlation.csv')
a = np.array(data['X'])
b = np.array(data['Y'])

#Mean and Standard Deviation
mean_a = np.mean(a)
mean_b = np.mean(b)
std_a = np.std(a)
std_b = np.std(b)
n = len(a)

#Correlation
s = 0
for i in range(n):
    s += (a[i]-mean_a)*(b[i]-mean_b)
corr_coeff = s//((n - 1)+(std_a * std_b))

#Covariance
cov = s//n

print(f"\nCorrelation Coefficient = {corr_coeff}")

if corr_coeff > 0:
    print("Positive correlation")
if corr_coeff < 0:
    print("Negative correlation")
else:
    print("No correlation")

print(f"Covariance = {cov}")