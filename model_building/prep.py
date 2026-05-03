# for data manipulation
import pandas as pd
import os
# for data preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# for hugging face authentication
from huggingface_hub import HfApi

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Load dataset directly from Hugging Face
DATASET_PATH = "hf://datasets/GopalSriram/tourism-package-prediction/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("✅ Dataset loaded successfully. Shape:", df.shape)

# Drop unnecessary column
df.drop(columns=['CustomerID'], inplace=True)

# Fill missing values with median
df.fillna(df.median(numeric_only=True), inplace=True)

# Encode categorical columns
label_encoder = LabelEncoder()
categorical_cols = ['TypeofContact', 'Occupation', 'Gender',
                    'MaritalStatus', 'Designation', 'ProductPitched']

for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col].astype(str))

print("✅ Encoding and cleaning done.")

# Define target column
target_col = 'ProdTaken'

# Split into features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("✅ Train shape:", Xtrain.shape)
print("✅ Test shape:", Xtest.shape)

# Save locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("✅ Train/test CSV files saved locally.")

# Upload back to Hugging Face dataset repo
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id="GopalSriram/tourism-package-prediction",
        repo_type="dataset",
    )
    print(f"✅ Uploaded: {file_path}")

print("✅ All train/test splits uploaded to Hugging Face.")
