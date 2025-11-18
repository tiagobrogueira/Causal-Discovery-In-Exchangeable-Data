import os
import shutil
import requests
from tqdm import tqdm

# --- Helper Functions ---

def _get_headers(token=None):
    """Returns headers, including Authorization if a GITHUB_TOKEN is found."""
    if token is None:
        token = os.getenv("GITHUB_TOKEN")
    if token:
        # Use Bearer token authorization
        return {"Authorization": f"Bearer {token}"}
    return {}


def _check_response_status(response):
    """Checks for non-2xx status codes and raises an exception."""
    # This replaces the complex _handle_rate_limit function
    response.raise_for_status()


def _download_file(url, local_path, token=None):
    """Download a single file with progress bar."""
    # Note: Authorization headers are typically ignored by raw.githubusercontent.com
    headers = _get_headers(token) 
    r = requests.get(url, headers=headers, stream=True)
    _check_response_status(r) # Check for HTTP errors

    total = int(r.headers.get("content-length", 0))
    # Ensure directory exists before opening file
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    with open(local_path, "wb") as f, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        desc=f"Downloading {os.path.basename(local_path)}",
        leave=False,
    ) as bar:
        for chunk in r.iter_content(1024):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

# --- Efficient Logic using Git Trees API ---

def _get_recursive_tree_files(owner_repo, base_path, token=None):
    """
    Fetches a recursive list of all files in the tree rooted at base_path 
    using a single Git Trees API call.
    """
    
    headers = _get_headers(token)
    
    # 1. Get the SHA of the default branch
    repo_url = f"https://api.github.com/repos/{owner_repo}"
    response = requests.get(repo_url, headers=headers)
    _check_response_status(response)
    default_branch = response.json().get('default_branch', 'main')

    # 2. Get the commit SHA for the default branch
    branch_url = f"https://api.github.com/repos/{owner_repo}/branches/{default_branch}"
    response = requests.get(branch_url, headers=headers)
    _check_response_status(response)
    tree_sha = response.json()['commit']['sha']
    
    # 3. Get the full recursive tree (ONE API call)
    tree_url = f"https://api.github.com/repos/{owner_repo}/git/trees/{tree_sha}?recursive=1"
    response = requests.get(tree_url, headers=headers)
    _check_response_status(response)
    
    full_tree = response.json().get("tree", [])
    
    # Filter the tree to include only the files ('blob') under the specified base_path
    prefix = base_path.strip('/') + '/'
    
    files_info = []
    for item in full_tree:
        if item["type"] == "blob" and item["path"].startswith(prefix):
            files_info.append({
                "repo_path": item["path"],
                # Use raw content URL for non-rate-limited downloads
                "download_url": f"https://raw.githubusercontent.com/{owner_repo}/{default_branch}/{item['path']}"
            })
            
    return files_info, default_branch


# --- Main Download Functions ---

def download_tuebingen(write_dir="benchmarks/Tuebingen",
                       repo="https://github.com/tiagobrogueira/Causal-Discovery-In-Exchangeable-Data",
                       overwrite=False,
                       token=None):
    
    # --- Setup ---
    if repo.endswith("/"):
        repo = repo[:-1]
    owner_repo = "/".join(repo.split("/")[-2:])
    target_path = "bicausal/benchmarks/Tuebingen"

    if os.path.exists(write_dir) and overwrite:
        shutil.rmtree(write_dir)
    os.makedirs(write_dir, exist_ok=True)
    
    # --- Download Logic ---
    try:
        files_info, _ = _get_recursive_tree_files(owner_repo, target_path, token=token)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching file list for Tuebingen: {e}")
        return

    total_files = len(files_info)
    
    with tqdm(total=total_files, desc="Downloading Tuebingen", unit="file") as pbar:
        for file_info in files_info:
            repo_path = file_info["repo_path"] 
            download_url = file_info["download_url"]
            
            relative_path = os.path.relpath(repo_path, start=target_path)
            local_path = os.path.join(write_dir, relative_path)
            
            if os.path.exists(local_path) and not overwrite:
                pbar.update(1)
                continue
            
            _download_file(download_url, local_path, token=token)
            pbar.update(1)

    print(f"[download_tuebingen] Download completed (incremental mode={not overwrite}): {write_dir}")


def download_lisbon(write_dir="benchmarks/Lisbon",
                    repo="https://github.com/tiagobrogueira/Causal-Discovery-In-Exchangeable-Data",
                    overwrite=False,
                    figures=False,
                    token=None):
    
    # --- Setup ---
    if repo.endswith("/"):
        repo = repo[:-1]
    owner_repo = "/".join(repo.split("/")[-2:])
    base_path = "bicausal/benchmarks/Lisbon"

    if os.path.exists(write_dir) and overwrite:
        shutil.rmtree(write_dir)
    os.makedirs(write_dir, exist_ok=True)

    # --- Download Logic ---
    try:
        files_info, _ = _get_recursive_tree_files(owner_repo, base_path, token=token)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching file list for Lisbon: {e}")
        return

    skip_path_prefix = f"{base_path}/pictures/" if not figures else None
    
    # Filter files locally based on the 'figures' flag
    files_to_download = []
    for file_info in files_info:
        repo_path = file_info["repo_path"]
        if skip_path_prefix and repo_path.startswith(skip_path_prefix):
            continue
        files_to_download.append(file_info)

    total_files = len(files_to_download)
    
    with tqdm(total=total_files, desc="Downloading Lisbon", unit="file") as pbar:
        for file_info in files_to_download:
            repo_path = file_info["repo_path"] 
            download_url = file_info["download_url"]
            
            relative_path = os.path.relpath(repo_path, start=base_path)
            local_path = os.path.join(write_dir, relative_path)
            
            if os.path.exists(local_path) and not overwrite:
                pbar.update(1)
                continue
            
            _download_file(download_url, local_path, token=token)
            pbar.update(1)

    print(f"[download_lisbon] Download completed (incremental mode={not overwrite}): {write_dir}")