"""Test metadata extraction from .rec files and file matching."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from hydro_analysis.core.io import DatasetIndex

test_path = Path(r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung\50 nm")

def main():
    idx = DatasetIndex.from_root(test_path)
    
    if not idx.datasets:
        print("No datasets found")
        return
    
    print(f"\nFound {len(idx.datasets)} datasets in: {test_path}\n")
    
    for i, base_name in enumerate(idx.list_bases(), 1):
        ds = idx.get(base_name)
        rec_meta = ds.rec_metadata
        
        print(f"{i}. {base_name}")
        print(f"   Base TIF: {ds.base_tif.name}")
        
        if ds.processed_tifs:
            print(f"   Processed TIFs ({len(ds.processed_tifs)}):")
            for tif in ds.processed_tifs:
                print(f"      - {tif.name}")
        
        print(f"   REC: {ds.rec_path.name if ds.rec_path else 'None'}")
        
        if rec_meta and 'error' not in rec_meta:
            if 'size_px' in rec_meta:
                width = rec_meta['size_px']['x']
                height = rec_meta['size_px']['y']
                fps = rec_meta.get('fps_nominal', 0)
                print(f"   Metadata: {width}x{height} px, {fps:.1f} fps")
                if 'exposure_ms' in rec_meta:
                    print(f"   Exposure: {rec_meta['exposure_ms']:.2f} ms")
        
        if ds.xml_paths:
            print(f"   XMLs ({len(ds.xml_paths)}):")
            for xml in ds.xml_paths:
                print(f"      - {xml.name}")
        else:
            print(f"   XMLs: NONE")
        print()
    
    with_xml = sum(1 for name in idx.list_bases() if idx.get(name).xml_paths)
    print(f"Summary: {len(idx.datasets)} datasets, {with_xml} have XML tracks")

if __name__ == "__main__":
    main()
