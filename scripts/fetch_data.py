import requests
import json
from pathlib import Path

def fetch_all_links(input_file="data/drupal_links.txt", output_file="data/raw_data.json"):
    all_data = []
    with open(input_file, "r") as f:
        urls = [line.strip() for line in f.readlines()]
    
    for url in urls:
        print(f"Fetching: {url}")
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                all_data.append({
                    "url": url,
                    "data": response.json()
                })
        except Exception as e:
            print(f"Error fetching {url}: {e}")
    
    Path("data").mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(all_data, f, indent=2)

if __name__ == "__main__":
    fetch_all_links()
