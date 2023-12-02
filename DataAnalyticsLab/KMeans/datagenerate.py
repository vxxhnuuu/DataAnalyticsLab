import numpy as np
import pandas as pd

# Generate a sample dataset
np.random.seed(42)
data = {
    'Feature1': np.random.rand(100),
    'Feature2': np.random.rand(100),
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('kmeans.csv', index=False)
