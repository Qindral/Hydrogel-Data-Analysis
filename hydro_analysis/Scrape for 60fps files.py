"""
Script to find and list TIFF files matching specific criteria.

Searches through directories for TIF files containing "20" and "nm" in filename,
excluding files with "mg", "A", or "B". Lists files with dimensions 200x150.
"""

from pathlib import Path
import tifffile
import re


def find_matching_tiffs(root_dir: str = "Data") -> list[Path]:
    """
    Search for TIFF files matching specific naming and dimension criteria.
    
    Args:
        root_dir: Root directory to start search (default: "Data")
    
    Returns:
        List of Path objects for matching files
    """
    root_path = Path(root_dir)
    matching_files = []
    
    # Recursively find all TIF/TIFF files
    tif_patterns = ["**/*.tif", "**/*.tiff", "**/*.TIF", "**/*.TIFF"]
    all_tifs = []
    for pattern in tif_patterns:
        all_tifs.extend(root_path.glob(pattern))
    
    print(f"Found {len(all_tifs)} TIFF files total")
    
    for tif_path in all_tifs:
        filename = tif_path.name
        
        # Check filename criteria
        has_20 = "50" in filename
        has_nm = "nm" in filename.lower()
        # Define forbidden terms
        forbidden_terms = ["mg", "A", "B","at","processed"]
        
        # Check if any forbidden term is in filename
        has_forbidden = any(term in filename for term in forbidden_terms)
        
        # Skip if doesn't match naming criteria
        if not (has_20 and has_nm):
            continue
        if has_forbidden:
            continue
        
        # Check dimensions
        try:
            with tifffile.TiffFile(tif_path) as tif:
                # Get first page dimensions
                page = tif.pages[0]
                height, width = page.shape[:2]
                
                if width == 200 and height == 150:
                    matching_files.append(tif_path)
                    print(f"✓ {tif_path.relative_to(root_path)} ({width}x{height})")
                    
        except Exception as e:
            print(f"✗ Could not read {tif_path.name}: {e}")
            continue
    
    return matching_files


def main():
    """Main execution."""

    
    matches = find_matching_tiffs(r'E:\PhD Data Analysis\SPT 2025 II')
    
    print()
    print("=" * 60)
    print(f"Found {len(matches)} matching files")
    print("=" * 60)
    
    if matches:
        print("\nMatching files:")
        for path in matches:
            print(f"  {path}")


if __name__ == "__main__":
    main()