from huggingface_hub import snapshot_download
snapshot_download(repo_id="osv5m/osv5m", local_dir="/root/data", repo_type='dataset')
