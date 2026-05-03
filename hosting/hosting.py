from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

api.upload_folder(
    folder_path="deployment",           # local deployment folder
    repo_id="GopalSriram/Tourism-Package-Prediction",  # your HF space
    repo_type="space",
    path_in_repo="",                    # upload to root of space
)
print("✅ Deployment files pushed to Hugging Face Space.")
