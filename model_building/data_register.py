from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

repo_id = "GopalSriram/tourism-package-prediction"
repo_type = "dataset"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the dataset repo exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Repo '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Repo '{repo_id}' not found. Creating new repo...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Repo '{repo_id}' created.")

# Step 2: Use path relative to THIS script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(script_dir, "..", "data")
data_folder = os.path.abspath(data_folder)

print(f"Uploading from: {data_folder}")

# Step 3: Upload the data folder to HF dataset repo
api.upload_folder(
    folder_path=data_folder,
    repo_id=repo_id,
    repo_type=repo_type,
)
print("✅ Data uploaded to Hugging Face dataset repo.")
