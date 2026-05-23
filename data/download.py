import urllib.request
import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

SOURCES = {
    "verdict.txt": "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt",
}


def download(filename: str) -> None:
    url = SOURCES[filename]
    dest = os.path.join(DATA_DIR, filename)
    if os.path.exists(dest):
        print(f"{filename} already exists, skipping.")
        return
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved to {dest}")


if __name__ == "__main__":
    for name in SOURCES:
        download(name)
