"""
This script confirms:
 - that the number of images in the dataset directory train/val matches those in the json folder
 - Also checks that the file names in the images directory match those in the json folder.
 - collects and prints out the misnomer in the image names with no matching json files or vice versa.

"""

import os
from pathlib import Path
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Confirm dataset count in directory.")
    parser.add_argument(
        "--data_root", type=Path, required=True, help="Path to the dataset directory."
    )
    parser.add_argument(
        "--ext", type=str, default="png", help="Image file extension (default: jpg)."
    )
    args = parser.parse_args()

    data_root = args.data_root
    images_dir = data_root / "images"
    json_dir = data_root / "compact"

    images_list = sorted(images_dir.glob(f"*.{args.ext.lower()}"))
    json_list = sorted(json_dir.glob("*.json"))
    images_count = len(images_list)
    json_count = len(json_list)
    print(f"Images count: {images_count}")
    print(f"JSON files count: {json_count}")
    if images_count != json_count:
        print("Mismatch in counts!")
        print(f"Images: {images_count}, JSONs: {json_count}")

    else:
        print("Counts match.")
    # Check for mismatches in file names
    images_set = {img.stem for img in images_list}
    json_set = {json.stem for json in json_list}
    missing_in_json = images_set - json_set
    missing_in_images = json_set - images_set
    if missing_in_json:
        print("Images with no matching JSON files:")
        for img in sorted(missing_in_json):
            print(f"  {img}.{args.ext}")
    else:
        print("All images have matching JSON files.")
    if missing_in_images:
        print("JSON files with no matching images:")
        for json in sorted(missing_in_images):
            print(f"  {json}.json")
    else:
        print("All JSON files have matching images.")
    # Print mismatches
    if missing_in_json or missing_in_images:
        print("\nMismatches found:")
        for img in sorted(missing_in_json):
            print(f"Image with no JSON: {img}.{args.ext}")
        for json in sorted(missing_in_images):
            print(f"JSON with no image: {json}.json")
    else:
        print("No mismatches found.")
    print("Dataset count confirmation completed.")
