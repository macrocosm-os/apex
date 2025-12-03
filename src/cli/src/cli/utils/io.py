from pathlib import Path


def get_folder_options(folder: Path) -> list[str]:
    if not folder.exists():
        return []
    return sorted([f.name for f in folder.iterdir() if f.is_dir() or f.is_file()])
