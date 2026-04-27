import os
from PIL import Image
import numpy as np

def _iter_class_directories(path):
  """Yield (split_name_or_none, class_name, class_dir) for supported dataset layouts."""
  class_names = ["NORMAL", "PNEUMONIA"]
  split_names = ["train", "test", "val"]

  # Layout A: path/NORMAL, path/PNEUMONIA
  flat_found = False
  for class_name in class_names:
    class_dir = os.path.join(path, class_name)
    if os.path.isdir(class_dir):
      flat_found = True
      yield None, class_name, class_dir

  if flat_found:
    return

  # Layout B: path/train|test|val/NORMAL|PNEUMONIA
  for split_name in split_names:
    split_dir = os.path.join(path, split_name)
    if not os.path.isdir(split_dir):
      continue
    for class_name in class_names:
      class_dir = os.path.join(split_dir, class_name)
      if os.path.isdir(class_dir):
        yield split_name, class_name, class_dir


# Normalize images and save locally
# Also check if images have alreaduy been normalized and saved, to avoid doing it again.
def normalize_and_save_images(path, output_path):
  # check if normalized images already exist
  if os.path.exists(output_path):
    print(f"Output path {output_path} already exists. Skipping normalization.")
    return

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
          normalized_image = normalize_image(image_path)
          output_image_path = os.path.join(output_subfolder, filename)
          Image.fromarray((normalized_image[0] * 255).astype(np.uint8)).save(output_image_path)
          saved_images += 1
          if saved_images % 100 == 0:
            print(f"Saved {saved_images} normalized images")
        except Exception as error:
          print(f"Skipping {image_path}: {error}")

  if saved_images == 0:
    print(
      "No images were saved. Expected either path/NORMAL|PNEUMONIA or "
      "path/train|test|val/NORMAL|PNEUMONIA"
    )
  else:
    print(f"Done. Saved {saved_images} normalized images to {output_path}")


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
  for _, class_name, class_dir in _iter_class_directories(path):
    if split_name is not None and not class_dir.startswith(os.path.join(path, split_name)):
      continue
    for filename in os.listdir(class_dir):
      image_path = os.path.join(class_dir, filename)
      if os.path.isfile(image_path):
        try:
          img_array = normalize_image(image_path)
          images.append(img_array)
          labels.append(0 if class_name == "NORMAL" else 1)
        except Exception as error:
          print(f"Skipping {image_path}: {error}")
  return np.array(images), np.array(labels)