import os

folder_path = "datasets/"  # e.g., "./data"
csv_files = [f.lower().split('.')[0] for f in os.listdir(folder_path)]

print(csv_files)