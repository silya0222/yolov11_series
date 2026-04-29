from __future__ import annotations

from pathlib import Path

import yaml
from clearml import Dataset


def prepare_yolo_data_yaml(
    dataset_project: str,
    dataset_name: str,
    dataset_version: str | None,
    template_yaml: str,
    output_yaml: str = "clearml_dataset.yaml",
) -> str:
    """Download/cache a ClearML dataset and create a YOLO data yaml for this machine."""
    dataset = Dataset.get(
        dataset_project=dataset_project,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        only_completed=True,
    )
    dataset_root = Path(dataset.get_local_copy()).resolve()

    with open(template_yaml, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    data_cfg["path"] = dataset_root.as_posix()

    output_path = Path(output_yaml).resolve()
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_cfg, f, allow_unicode=True, sort_keys=False)

    print(f"[ClearML Data] Dataset root: {dataset_root}")
    print(f"[ClearML Data] YOLO data yaml: {output_path}")
    return str(output_path)
