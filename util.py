import os
from PIL import Image
import numpy as np


def _canonicalize_split_name(split_name):
  """Map split aliases to canonical names used by this project."""
  if split_name is None:
    return None
  normalized = str(split_name).strip().lower()
  mapping = {
    "train": "train",
    "training": "train",
    "test": "test",
    "testing": "test",
    "eval": "eval",
    "evaluation": "eval",
    "val": "eval",
    "validation": "eval",
  }
  return mapping.get(normalized, normalized)


def _canonicalize_class_name(class_name):
  """Map class aliases to canonical binary labels used by this project."""
  normalized = str(class_name).strip().lower()
  mapping = {
    "normal": "NORMAL",
    "pneumonia": "PNEUMONIA",
  }
  return mapping.get(normalized)

def _iter_class_directories(path):
  """Yield (split_name_or_none, class_name, class_dir) for supported dataset layouts."""
  split_aliases = {
    "train": ["train", "training"],
    "test": ["test", "testing"],
    "eval": ["eval", "val", "validation", "valid"],
  }

  # Layout A: path/NORMAL|normal, path/PNEUMONIA|pneumonia
  flat_found = False
  for entry in os.listdir(path):
    class_dir = os.path.join(path, entry)
    if not os.path.isdir(class_dir):
      continue
    class_name = _canonicalize_class_name(entry)
    if class_name is None:
      continue
    flat_found = True
    yield None, class_name, class_dir

  if flat_found:
    return

  # Layout B: path/train|test|eval|val/NORMAL|PNEUMONIA
  for split_name, aliases in split_aliases.items():
    split_dir = None
    for alias in aliases:
      candidate = os.path.join(path, alias)
      if os.path.isdir(candidate):
        split_dir = candidate
        break

    if split_dir is None:
      continue

    for entry in os.listdir(split_dir):
      class_dir = os.path.join(split_dir, entry)
      if not os.path.isdir(class_dir):
        continue
      class_name = _canonicalize_class_name(entry)
      if class_name is None:
        continue
      yield split_name, class_name, class_dir


# Normalize images and save locally
# Also check if images have alreaduy been normalized and saved, to avoid doing it again.
def normalize_and_save_images(path, output_path):
  # Normalize incrementally so missing splits/classes are still added.
  existing_files = 0
  if os.path.exists(output_path):
    print(f"Output path {output_path} already exists. Normalizing only missing files.")

  saved_images = 0

  for split_name, class_name, class_dir in _iter_class_directories(path):
    if split_name is None:
      output_subfolder = os.path.join(output_path, class_name)
    else:
      output_subfolder = os.path.join(output_path, split_name, class_name)

    os.makedirs(output_subfolder, exist_ok=True)

    for filename in os.listdir(class_dir):
      image_path = os.path.join(class_dir, filename)
      if os.path.isfile(image_path):
        try:
          output_image_path = os.path.join(output_subfolder, filename)
          if os.path.exists(output_image_path):
            existing_files += 1
            continue

          normalized_image = normalize_image(image_path)
          Image.fromarray((normalized_image[0] * 255).astype(np.uint8)).save(output_image_path)
          saved_images += 1
          if saved_images % 100 == 0:
            print(f"Saved {saved_images} normalized images")
        except Exception as error:
          print(f"Skipping {image_path}: {error}")

  if saved_images == 0:
    print(
      "No images were saved. Expected either path/NORMAL|PNEUMONIA or "
      "path/train|test|eval|val/NORMAL|PNEUMONIA"
    )
  else:
    print(f"Done. Saved {saved_images} normalized images to {output_path}")

  if existing_files > 0:
    print(f"Skipped {existing_files} files that were already normalized")


def normalize_image(image):
  with Image.open(image) as img:
    img = img.resize((224, 224), Image.LANCZOS)  # Resize to 224x224
    img = img.convert("L")
    # Use float32 to cut memory usage in half versus float64.
    img_array = np.asarray(img, dtype=np.float32) / 255.0
  return img_array.reshape(1, 224, 224)


def load_dataset(path, split_name):
  images = []
  labels = []
  target_split = _canonicalize_split_name(split_name)
  matched_class_dirs = 0

  for found_split, class_name, class_dir in _iter_class_directories(path):
    if target_split is not None and found_split != target_split:
      continue
    matched_class_dirs += 1
    for filename in os.listdir(class_dir):
      image_path = os.path.join(class_dir, filename)
      if os.path.isfile(image_path):
        try:
          img_array = normalize_image(image_path)
          images.append(img_array)
          labels.append(0 if class_name == "NORMAL" else 1)
        except Exception as error:
          print(f"Skipping {image_path}: {error}")

  if target_split is not None and matched_class_dirs == 0:
    available_splits = sorted(
      {
        found_split
        for found_split, _, _ in _iter_class_directories(path)
        if found_split is not None
      }
    )
    raise ValueError(
      f"No class directories found for split '{split_name}' in '{path}'. "
      f"Available canonical splits: {available_splits or 'none'}"
    )

  if len(images) == 0:
    split_label = target_split if target_split is not None else "all"
    raise ValueError(
      f"Loaded 0 images for split '{split_label}' from '{path}'. "
      "Check that the directory contains readable image files."
    )

  return np.array(images), np.array(labels)