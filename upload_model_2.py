import os
from huggingface_hub import HfApi
from tqdm.auto import tqdm # For a nice progress bar

# --- Configuration ---
# 1. The local path to your sharded checkpoint folder.
local_folder_path = "trainer_output/checkpoint-4347"

# 2. Your Hugging Face repository ID.
repo_id = "Thermostatic/neuraltranslate-27b-mt-es-nah-v1.2"

# --- Prerequisites ---
# - Make sure you are logged in:
#   In your terminal, run: `huggingface-cli login`
#   and provide a token with 'write' permissions.
# - Install tqdm for the progress bar: `pip install tqdm`

# --- Main Upload Logic ---
print(f"Starting upload of files from '{local_folder_path}' to '{repo_id}'")

# Check if the local directory exists
if not os.path.isdir(local_folder_path):
    raise FileNotFoundError(
        f"The specified local folder path does not exist: {local_folder_path}"
    )

# Initialize the HfApi client
api = HfApi()

# Get a list of all files to upload
# os.walk is robust and handles files in subdirectories, though you likely have a flat structure.
files_to_upload = []
for root, _, files in os.walk(local_folder_path):
    for filename in files:
        # Create the full local path and the desired path in the repo
        files_to_upload.append({
            "local_path": os.path.join(root, filename),
            "repo_path": os.path.relpath(os.path.join(root, filename), local_folder_path)
        })

if not files_to_upload:
    print("No files found in the specified directory. Exiting.")
    exit()

print(f"Found {len(files_to_upload)} files to upload.")

# Create the repo if it doesn't exist (and we have permission)
api.create_repo(repo_id, repo_type="model", exist_ok=True)

# Loop through the files and upload each one individually
# Using tqdm will give us a nice progress bar for the number of files
for file_info in tqdm(files_to_upload, desc="Uploading files"):
    local_path = file_info["local_path"]
    repo_path = file_info["repo_path"]
    
    # Generate a specific commit message for each file
    commit_msg = f"Upload {os.path.basename(repo_path)}"
    
    print(f"\nUploading {local_path} to {repo_path} in repo {repo_id}...")
    
    try:
        # The upload_file function is designed for large files and is resumable.
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_msg,
        )
        print(f"✅ Successfully uploaded {os.path.basename(repo_path)}")
    except Exception as e:
        print(f"❌ Failed to upload {os.path.basename(repo_path)}: {e}")
        print("Skipping this file and continuing with the next one.")

print("\n---")
print("✅ All files have been processed.")
print(f"Check your repository at: https://huggingface.co/{repo_id}/tree/main")