import os
from tqdm import tqdm

if __name__ == "__main__":
  new_dir = "data"
  if not os.path.isdir(new_dir):
    os.mkdir("data")

  for dir in ["dev", "test", "train"]:
    with open(os.path.join(new_dir, dir + ".json"), "w") as saved_file:
      saved_file.write("")
    saved_file = open(os.path.join(new_dir, dir + ".json"), "a")
    for filename in tqdm(os.listdir(dir)):
      if filename.startswith("."): continue
      with open(os.path.join(dir, filename), "r") as file:
        saved_file.write(file.readline() + "\n")
    saved_file.close()
