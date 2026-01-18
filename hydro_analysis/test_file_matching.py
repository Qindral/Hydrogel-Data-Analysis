"""Test the new file matching system."""

from pathlib import Path
from core import find_dataset_files

def test_file_matching():
    """Test file discovery and matching."""
    
    test_path = Path(r"C:\Users\Jonas\Documents\GitHub\Hydrogel-Data-Analysis\Data\29Aug_lang\TileScan 2\B\2")
    
    if not test_path.exists():
        print(f"Test path not found: {test_path}")
        return
    
    print("Testing file matching system")
    print("="*70)
    print(f"Path: {test_path}\n")
    
    datasets = find_dataset_files(test_path)
    
    print(f"Found {len(datasets)} datasets:\n")
    
    for i, ds in enumerate(datasets, 1):
        print(f"{i}. {ds.base_name}")
        print(f"   XML: {ds.xml_path.name} {'✓' if ds.xml_path.exists() else '✗'}")
        print(f"   TIF: {ds.tif_path.name if ds.tif_path else 'NOT FOUND'} {'✓' if ds.tif_path and ds.tif_path.exists() else '✗' if ds.tif_path else '-'}")
        print(f"   REC: {ds.rec_path.name if ds.rec_path else 'NOT FOUND'} {'✓' if ds.rec_path and ds.rec_path.exists() else '✗' if ds.rec_path else '-'}")
        print()
    
    # Summary
    matched_tif = sum(1 for ds in datasets if ds.tif_path)
    matched_rec = sum(1 for ds in datasets if ds.rec_path)
    
    print("="*70)
    print(f"Summary: {len(datasets)} XML files")
    print(f"         {matched_tif} matched TIF files")
    print(f"         {matched_rec} matched REC files")

if __name__ == "__main__":
    test_file_matching()
