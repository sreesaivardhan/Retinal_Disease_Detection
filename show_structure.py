import os
import json

# Use your actual dataset path - replace with where you actually put the datasets
DATASET_PATH = 'D:\\Retinal_Disease_Detection\\data\\raw'  # Current directory or specify exact path

structure = {}
for root, dirs, files in os.walk(DATASET_PATH):
    if any(keyword in root.lower() for keyword in ['aptos', 'rfmid', 'oct5k', 'mured']):
        path = os.path.relpath(root, DATASET_PATH)
        structure[path] = {
            'dirs': dirs[:10], 
            'files': [f for f in files[:10] if f.endswith(('.csv', '.png', '.jpg', '.jpeg'))]
        }

with open('actual_dataset_structure.json', 'w') as f:
    json.dump(structure, f, indent=4)

print(f'Found {len(structure)} relevant folders')
print('Structure saved to actual_dataset_structure.json')
