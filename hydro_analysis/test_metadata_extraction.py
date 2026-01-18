"""Test metadata extraction from .rec files and file matching."""

from pathlib import Path
import pandas as pd
from core import find_dataset_files
from core.io import parse_rec_file, get_mpp_from_dimensions

test_path = Path(r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung\50 nm")

def main():
    datasets = find_dataset_files(test_path)
    
    if not datasets:
        print("No TIF files found")
        return
    
    print(f"\nFound {len(datasets)} datasets in: {test_path}\n")
    
    for i, ds in enumerate(datasets, 1):
        rec_meta = parse_rec_file(ds.rec_path)
        
        print(f"{i}. {ds.base_name}")
        print(f"   Base TIF: {ds.base_tif.name}")
        
        if ds.processed_tifs:
            print(f"   Processed TIFs ({len(ds.processed_tifs)}):")
            for tif in ds.processed_tifs:
                print(f"      - {tif.name}")
        
        print(f"   REC: {ds.rec_path.name}")
        
        if rec_meta:
            if 'width' in rec_meta and 'height' in rec_meta:
                mpp = get_mpp_from_dimensions(rec_meta['width'], rec_meta['height'])
                fps = rec_meta.get('fps', 0)
                print(f"   Metadata: {rec_meta['width']}x{rec_meta['height']} px, {mpp} µm/px, {fps:.1f} fps")
        
        if ds.xml_paths:
            print(f"   XMLs ({len(ds.xml_paths)}):")
            for xml in ds.xml_paths:
                print(f"      - {xml.name}")
        else:
            print(f"   XMLs: NONE")
        print()
    
    with_xml = sum(1 for ds in datasets if ds.xml_paths)
    print(f"Summary: {len(datasets)} datasets, {with_xml} have XML tracks")

if __name__ == "__main__":
    main()
