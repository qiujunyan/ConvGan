from word_correction import spelling_correcter
import os
import json
from tqdm import tqdm

if __name__ == "__main__":
    data_dirs = ["dev", "train", "test"]
    root_dir = "./data/mutual/"
    for data_dir in data_dirs:
      file_dir = os.path.join(root_dir, data_dir)
      for file in tqdm(os.listdir(file_dir)):
        with open(os.path.join(file_dir, file), "r") as f:
          contents = json.load(f)

          a = 1