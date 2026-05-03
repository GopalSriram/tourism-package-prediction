from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

repo_id = "GopalSriram/tourism-package-prediction"
repo_type = "dataset"

api = HfApi(token=os.getenv("HF_TOKEN"))

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Repo '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Repo '{repo_id}' created.")

api.upload_folder(
    folder_path="/content/tourism-package-prediction/data",  # ← absolute path, no confusion
    repo_id=repo_id,
    repo_type=repo_type,
)
print("✅ Data uploaded to Hugging Face dataset repo.")
