"""小型文件工具：JSON/检查点原子替换，CSV以稳定列顺序写出。"""

import csv
import hashlib
import json
from pathlib import Path
import shutil
import time


def replace_with_retry(temporary, path):
    """Windows 短暂占用文件时重试替换；始终保留旧文件，持续失败则报错。"""
    for attempt in range(10):
        try:
            Path(temporary).replace(path)
            return
        except PermissionError as error:
            if getattr(error, "winerror", None) not in (5, 32, 33) or attempt == 9:
                raise
            time.sleep(min(0.1 * 2**attempt, 1.0))


def copy_atomic(source, destination):
    """best 也先写临时文件，避免复制中断破坏已有最佳权重。"""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".copy.tmp")
    shutil.copyfile(source, temporary)
    replace_with_retry(temporary, destination)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    replace_with_retry(temporary, path)


def write_csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    replace_with_retry(temporary, path)


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def append_csv(path, row, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()
