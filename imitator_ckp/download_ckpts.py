from huggingface_hub import snapshot_download

# Download all files matching the pattern to a local folder
snapshot_download(
    repo_id="LiKailin/ManipTrans",
    allow_patterns="imitator_ckp/*.pth",
    local_dir="."  # Download to the current directory
)

print("All .pth files from imitator_ckp downloaded.")