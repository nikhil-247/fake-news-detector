import pandas as pd

# Load the datasets
fake = pd.read_csv("fake-news-detector\Fake.csv")
real = pd.read_csv("fake-news-detector\True.csv")

# Add 'label' column
fake['label'] = 'FAKE'
real['label'] = 'REAL'

# Combine the datasets
combined = pd.concat([fake, real], ignore_index=True)

# Shuffle the data
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the result to a new CSV
combined.to_csv("fake-news-detector/data.csv", index=False)

print("✅ Combined dataset created as 'data.csv' with labels.")
