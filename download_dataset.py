import os
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# Replace with your Zenodo access token
ACCESS_TOKEN = "0ee7c29576f9f94f2b5515fa2541732118882d12"
if ACCESS_TOKEN == "":
    try:
        with open("token.txt", "r") as token_file:
            ACCESS_TOKEN = token_file.read().strip()
    except FileNotFoundError:
        print("Token file 'token.txt' not found. Please create the file and add your Zenodo access token or set the ACCESS_TOKEN variable in the script.")
        exit()
    exit()

record_id = "14223624" #LUNA25 record id

# Specify the output folder where files will be saved
output_folder = "/vol/csedu-nobackup/course/IMC037_aimi/group07/"
if output_folder == "":
    print("Please set your output folder in the script.")
    exit()

os.makedirs(output_folder, exist_ok=True)

# Get the metadata of the Zenodo record
r = requests.get(f"https://zenodo.org/api/records/{record_id}", params={'access_token': ACCESS_TOKEN})

if r.status_code != 200:
    print("Error retrieving record:", r.status_code, r.text)
    exit()

# Extract download URLs and filenames
download_urls = [f['links']['self'] for f in r.json()['files']]
filenames = [f['key'] for f in r.json()['files']]

print(f"Total files to download: {len(download_urls)}")

# Get metadata
r.raise_for_status()
files = r.json()['files']
download_tasks = [(f['key'], f['links']['self']) for f in files]

def download_file(filename_url):
    filename, url = filename_url
    file_path = os.path.join(output_folder, filename)

    if os.path.exists(file_path):
        return f"{filename} already exists. Skipped."

    try:
        with requests.get(url, params={'access_token': ACCESS_TOKEN}, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(file_path, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                leave=False
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
        return f"{filename} downloaded successfully."
    except Exception as e:
        return f"Failed to download {filename}: {e}"

# Use ThreadPoolExecutor for I/O-bound concurrency
num_cpus = multiprocessing.cpu_count()
max_workers = min(num_cpus, len(download_tasks))  # Tune as needed
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(download_file, task) for task in download_tasks]
    for future in as_completed(futures):
        print(future.result())

print("All downloads completed (or skipped)!")