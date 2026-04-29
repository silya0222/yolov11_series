from __future__ import annotations

import argparse
from pathlib import Path

from clearml import Dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a YOLO dataset folder to ClearML Data.")
    parser.add_argument("--project", default="YOLOv11-Datasets", help="ClearML dataset project name.")
    parser.add_argument("--name", default="km_yolo_dataset", help="ClearML dataset name.")
    parser.add_argument("--version", default="v1", help="ClearML dataset version.")
    parser.add_argument("--folder", default="dataset/km", help="Local dataset root folder.")
    parser.add_argument(
        "--include",
        nargs="+",
        default=["*.jpg", "*.jpeg", "*.png", "*.txt", "*.yaml", "*.yml"],
        help="File patterns to upload. Cache files are intentionally excluded by default.",
    )
    parser.add_argument("--output-uri", default=None, help="Optional storage URI, e.g. s3://bucket/path.")
    parser.add_argument("--description", default="YOLO dataset uploaded from local project.")
    args = parser.parse_args()

    if len(args.name.strip()) < 3:
        raise ValueError("ClearML dataset name must be at least 3 characters. Use e.g. km_yolo_dataset.")

    folder = Path(args.folder).resolve()
    if not folder.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {folder}")

    dataset = Dataset.create(
        dataset_project=args.project,
        dataset_name=args.name,
        dataset_version=args.version,
        output_uri=args.output_uri,
        description=args.description,
    )
    added = dataset.add_files(
        path=str(folder),
        wildcard=args.include,
        local_base_folder=str(folder),
        dataset_path=".",
        recursive=True,
        verbose=True,
    )
    print(f"Added {added} files from {folder}")
    dataset.upload(show_progress=True, verbose=True)
    dataset.finalize(verbose=True)
    print(f"ClearML dataset id: {dataset.id}")
    print(f"Dataset: {args.project}/{args.name}:{args.version}")


if __name__ == "__main__":
    main()
