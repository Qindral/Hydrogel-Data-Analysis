"""Test script for the example folder structure."""

from pathlib import Path
import sys


# Test with the example folder structure
test_folder = Path(r"E:\PhD Data Analysis\SPT 2025 II\2026.01.19")

def print_folder_structure(folder: Path, indent: int = 0):
    """Recursively print folder structure with files."""
    prefix = "  " * indent
    
    if not folder.exists():
        print(f"Error: Folder does not exist: {folder}")
        return
    
    print(f"{prefix}{folder.name}/")
    
    try:
        items = sorted(folder.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        
        for item in items:
            if item.is_dir():
                print_folder_structure(item, indent + 1)
            else:
                print(f"{prefix}  {item.name}")
    except PermissionError:
        print(f"{prefix}  [Permission Denied]")

print_folder_structure(test_folder)