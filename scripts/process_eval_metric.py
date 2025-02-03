#%%
import pandas as pd

# Define the file path
file_path = '/jackal_ws/src/mlda-barn-2024/imit_out.txt'

# Read the data into a DataFrame
df = pd.read_csv(file_path, sep=" ", header=None)

# Filter out rows where the last column value is zero
filtered_df = df[df.iloc[:, -1] != 0]

# Calculate the average of the last column values that are not zero
average = filtered_df.iloc[:, -1].mean()

print("Average of the last column values that are not zero:", average)
# %%
