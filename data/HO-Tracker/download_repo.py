from huggingface_hub import snapshot_download
import os

REPO_ID = "LiKailin/HO-Tracker"

def download_entire_repo(repo_id, local_dir=None):
    """
    Downloads all files from a Hugging Face dataset repository
    to a specified local directory.
    """

    if local_dir is not None:
        # Ensure the target directory exists
        os.makedirs(local_dir, exist_ok=True)
    else:
        local_dir = "."

    print(f"Downloading repository {repo_id} to {local_dir}...")

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            repo_type="dataset",  # Specify "dataset"
            local_dir_use_symlinks=False # Avoid symlinks, download files directly
        )
        print("Download complete.")
        print(f"All files are saved in: {os.path.abspath(local_dir)}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    download_entire_repo(REPO_ID, "data/HO-Tracker")
