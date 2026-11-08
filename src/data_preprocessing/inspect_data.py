from pathlib import Path

# Define the project root directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Define data directories
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Create directories if they do not exist
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Display project directory paths
print("=== Project Paths ===")
print("Project:", BASE_DIR)
print("Raw data:", RAW_DIR)
print("Processed data:", PROCESSED_DIR)

# List raw data files
print("\n=== Raw Data Files ===")

files = sorted(
    file for file in RAW_DIR.iterdir()
    if file.is_file()
)

# Display file names and sizes
if not files:
    print("No data files found.")
else:
    for file in files:
        size_mb = file.stat().st_size / (1024 ** 2)
        print(f"{file.name} | {size_mb:.2f} MB")

print("\nStep 1 Completed.")
