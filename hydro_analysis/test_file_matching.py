"""Test the new DatasetIndex file matching system."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from hydro_analysis.core.io import DatasetIndex

def test_file_matching():
    """Test file discovery and matching with DatasetIndex."""
    
    test_path = Path(r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung\50 nm")
    
    if not test_path.exists():
        print(f"Test path not found: {test_path}")
        return
    
    print("Testing DatasetIndex file matching")
    print("="*70)
    print(f"Path: {test_path}\n")
    
    # Build index
    idx = DatasetIndex.from_root(test_path)
    
    print(f"Found {len(idx.datasets)} datasets:\n")
    
    for i, base_name in enumerate(idx.list_bases(), 1):
        ds = idx.get(base_name)
        print(f"{i}. {base_name}")
        print(f"   TIF: {ds.base_tif.name} {'✓' if ds.base_tif.exists() else '✗'}")
        print(f"   REC: {ds.rec_path.name if ds.rec_path else 'NOT FOUND'} {'✓' if ds.rec_path and ds.rec_path.exists() else '✗' if ds.rec_path else '-'}")
        print(f"   XMLs: {len(ds.xml_paths)}")
        for xml in ds.xml_paths:
            print(f"      - {xml.name} {'✓' if xml.exists() else '✗'}")
        if ds.processed_tifs:
            print(f"   Processed TIFs: {len(ds.processed_tifs)}")
            for p_tif in ds.processed_tifs:
                print(f"      - {p_tif.name} {'✓' if p_tif.exists() else '✗'}")
        print()
    
    # Summary
    matched_rec = sum(1 for name in idx.list_bases() if idx.get(name).rec_path)
    has_processed = sum(1 for name in idx.list_bases() if idx.get(name).processed_tifs)
    
    print("="*70)
    print(f"Summary: {len(idx.datasets)} datasets")
    print(f"         {matched_rec} with REC files")
    print(f"         {has_processed} with processed versions")

if __name__ == "__main__":
    test_file_matching()
