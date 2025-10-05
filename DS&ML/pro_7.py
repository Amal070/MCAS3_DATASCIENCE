# Import pandas
import pandas as pd

# Step 1: Read diabetes dataset
df = pd.read_csv("diabetes.csv")

# Step 2: Display first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Step 3: Define function to categorize BMI
def categorize_bmi(bmi):
    if bmi < -0.02:
        return "Low"
    elif -0.02 <= bmi <= 0.03:
        return "Medium"
    else:
        return "High"

# Step 4: Apply function to create new column
df["bmi_category"] = df["bmi"].apply(categorize_bmi)

# Step 5: Retrieve only 'bmi' and 'bmi_category' columns
result = df[["bmi", "bmi_category"]]

# Step 6: Display first 10 rows of result
print("\nBMI and BMI Category (first 10 rows):")
print(result.head(10))
