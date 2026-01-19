"""Test script for unified data loading from core.io module.

Demonstrates the new DatasetIndex API:
1. Build index from root folder
2. List all datasets (canonical base names)
3. Access dataset files by base name or any path
4. Load metadata from REC files
"""

from pathlib import Path
import sys
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hydro_analysis.core.io import DatasetIndex, DatasetFiles

def scrape_all_files(folder: str):
    """Recursively print folder structure with all files."""
    folder_path = Path(folder)
    
    if not folder_path.exists():
        print(f"Folder not found: {folder_path}")
        return
    
    print(f"\nFolder structure: {folder_path}")
    print("=" * 80)
    
    # Walk through directory tree
    for item in sorted(folder_path.rglob("*")):
        if item.is_file():
            # Calculate indentation based on depth
            depth = len(item.relative_to(folder_path).parts) - 1
            indent = "   " * depth
            print(f"{indent}File: {item.name}")
        elif item.is_dir():
            depth = len(item.relative_to(folder_path).parts) - 1
            indent = "   " * depth
            print(f"{indent}Dir: {item.name}/")

def test_find_datasets():
    """Test finding datasets in a folder."""


def test_build_index():
    """Test building dataset index from root folder."""
    print("\n" + "="*80)
    print("TEST 1: Building dataset index")
    print("="*80)
    
    test_folder = Path(r"E:\PhD Data Analysis\SPT 2025 II\2025.10.01")
    
    if not test_folder.exists():
        print(f"Test folder not found: {test_folder}")
        return
    
    # Build index
    idx = DatasetIndex.from_root(test_folder)
    
    print(f"\nFound {len(idx.datasets)} datasets:")
    for base_name in idx.list_bases()[:10]:  # Show first 10
        ds = idx.get(base_name)
        print(f"\n  Dataset: {base_name}")
        print(f"     Base TIF: {ds.base_tif.name}")
        print(f"     REC: {ds.rec_path.name if ds.rec_path else 'None'}")
        print(f"     Processed TIFs: {len(ds.processed_tifs)}")
        print(f"     XML files: {len(ds.xml_paths)}")
        for xml in ds.xml_paths[:3]:  # Show first 3 XMLs
            print(f"        - {xml.name}")
        if len(ds.xml_paths) > 3:
            print(f"        ... and {len(ds.xml_paths) - 3} more")


def test_rec_metadata():
    """Test extracting metadata from REC files."""
    print("\n" + "="*80)
    print("TEST 2: REC metadata extraction")
    print("="*80)
    
    test_folder = Path(r"E:\PhD Data Analysis\SPT 2025 II\2025.10.01")
    
    if not test_folder.exists():
        print(f"⚠️  Test folder not found: {test_folder}")
        return
    
    idx = DatasetIndex.from_root(test_folder)
    
    # Get first dataset with REC
    for base_name in idx.list_bases():
        ds = idx.get(base_name)
        if ds.rec_path:
            print(f"\nDataset: {base_name}")
            print(f"REC file: {ds.rec_path.name}")
            
            meta = ds.rec_metadata
            if meta and 'error' not in meta:
                print(f"\n✅ Metadata extracted:")
                print(f"   Camera: {meta.get('camera_type', 'N/A')}")
                print(f"   Size: {meta.get('size_px', 'N/A')}")
                print(f"   Exposure: {meta.get('exposure_ms', 'N/A')} ms")
                print(f"   Delay: {meta.get('delay_ms', 'N/A')} ms")
                print(f"   Frame period: {meta.get('frame_period_ms_nominal', 'N/A')} ms")
                print(f"   FPS: {meta.get('fps_nominal', 'N/A'):.2f}")
            else:
                print(f"❌ Failed to extract metadata: {meta.get('error', 'Unknown')}")
            break


def test_xml_groups():
    """Test XML grouping by variant (base vs processed)."""
    print("\n" + "="*80)
    print("TEST 3: XML grouping by variant")
    print("="*80)
    
    test_folder = Path(r"E:\PhD Data Analysis\SPT 2025 II\2025.10.01")
    
    if not test_folder.exists():
        print(f"⚠️  Test folder not found: {test_folder}")
        return
    
    idx = DatasetIndex.from_root(test_folder)
    
    # Show XML groups for first few datasets
    for base_name in idx.list_bases()[:5]:
        ds = idx.get(base_name)
        if ds.xml_paths:
            print(f"\nDataset: {base_name}")
            print(f"   Base XMLs: {len(ds.base_xmls)}")
            for xml in ds.base_xmls:
                print(f"      - {xml.name}")
            print(f"   Processed XMLs: {len(ds.processed_xmls)}")
            for xml in ds.processed_xmls:
                print(f"      - {xml.name}")


def test_path_resolution():
    """Test resolving dataset from any file path."""
    print("\n" + "="*80)
    print("TEST 4: Path resolution")
    print("="*80)
    
    test_folder = Path(r"E:\PhD Data Analysis\SPT 2025 II\2025.10.01")
    
    if not test_folder.exists():
        print(f"⚠️  Test folder not found: {test_folder}")
        return
    
    idx = DatasetIndex.from_root(test_folder)
    
    # Get first dataset and test resolution
    if not idx.list_bases():
        print("No datasets found")
        return
    
    base_name = idx.list_bases()[0]
    ds = idx.get(base_name)
    
    print(f"\nTesting resolution for dataset: {base_name}")
    
    # Test TIF resolution
    print(f"\n1. Resolving from base TIF: {ds.base_tif.name}")
    resolved = idx.from_any_path(ds.base_tif)
    print(f"   ✅ Resolved to: {resolved.base_name}")
    
    # Test REC resolution
    if ds.rec_path:
        print(f"\n2. Resolving from REC: {ds.rec_path.name}")
        resolved = idx.from_any_path(ds.rec_path)
        print(f"   ✅ Resolved to: {resolved.base_name}")
    
    # Test XML resolution
    if ds.xml_paths:
        print(f"\n3. Resolving from XML: {ds.xml_paths[0].name}")
        resolved = idx.from_any_path(ds.xml_paths[0])
        print(f"   ✅ Resolved to: {resolved.base_name}")


def test_processed_tifs():
    """Test finding processed TIF variants."""
    print("\n" + "="*80)
    print("TEST 5: Processed TIF detection")
    print("="*80)
    
    test_folder = Path(r"E:\PhD Data Analysis\SPT 2025 II\2025.10.01")
    
    if not test_folder.exists():
        print(f"⚠️  Test folder not found: {test_folder}")
        return
    
    idx = DatasetIndex.from_root(test_folder)
    
    # Find datasets with processed versions
    datasets_with_processed = [
        (name, idx.get(name)) for name in idx.list_bases()
        if len(idx.get(name).processed_tifs) > 0
    ]
    
    if not datasets_with_processed:
        print("No datasets with processed TIFs found")
        return
    
    print(f"\n✅ Found {len(datasets_with_processed)} datasets with processed versions:")
    
    for name, ds in datasets_with_processed[:5]:  # Show first 5
        print(f"\nDataset: {name}")
        print(f"   Base TIF: {ds.base_tif.name}")
        print(f"   Processed TIFs: {len(ds.processed_tifs)}")
        for p_tif in ds.processed_tifs:
            print(f"      - {p_tif.name}")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("DATASET INDEX TESTS")
    print("="*80)
    
    try:
        test_build_index()
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
    
    try:
        test_rec_metadata()
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
    
    try:
        test_xml_groups()
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
    
    try:
        test_path_resolution()
    except Exception as e:
        print(f"❌ Test 4 failed: {e}")
    
    try:
        test_processed_tifs()
    except Exception as e:
        print(f"❌ Test 5 failed: {e}")
    
    print("\n" + "="*80)
    print("TESTS COMPLETED")
    print("="*80)


if __name__ == "__main__":
    main()
