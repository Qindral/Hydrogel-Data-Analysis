"""Test script for grouped XML functionality.

Demonstrates how XMLs are now grouped by suffix for each base TIF file.

Example structure:
  50 nm_4.tif (base file)
    - XMLs grouped as:
      - 'base': 50 nm_4_Tracks.xml, 50nm_4.xml
      - 'processed': 50 nm_4_processed_Tracks.xml, 50nm_4_processed.xml
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from hydro_analysis.core.io import DatasetIndex

print("="*80)
print("TEST: Grouped XML Mapping by Suffix")
print("="*80)

test_folder = Path("Data")

if test_folder.exists():
    idx = DatasetIndex.from_root(test_folder)
    
    if idx.datasets:
        print(f"\n✅ Found {len(idx.datasets)} datasets\n")
        
        # Show first 5 datasets with grouped XMLs
        for i, base_name in enumerate(list(idx.list_bases())[:5], 1):
            ds = idx.get(base_name)
            print(f"{'='*70}")
            print(f"Dataset {i}: {base_name}")
            print(f"{'='*70}")
            print(f"📄 Base TIF: {ds.base_tif.name}")
            print(f"📂 Location: {ds.base_tif.parent}")
            print(f"🎚️  REC file: {ds.rec_path.name if ds.rec_path else 'None'}")
            
            # Show all XMLs (backward compatible)
            print(f"\n📋 All XMLs ({len(ds.xml_paths)}):")
            for xml in ds.xml_paths:
                print(f"   - {xml.name}")
            
            # Show grouped XMLs (NEW!)
            print(f"\n🗂️  Grouped XMLs:")
            for suffix, xmls in sorted(ds.xml_groups.items()):
                print(f"   [{suffix}] ({len(xmls)} files):")
                for xml in xmls:
                    print(f"      - {xml.name}")
            
            # Show convenient properties
            print(f"\n🎯 Quick Access:")
            print(f"   Base XMLs: {len(ds.base_xmls)} files")
            for xml in ds.base_xmls:
                print(f"      - {xml.name}")
            
            if ds.processed_xmls:
                print(f"   Processed XMLs: {len(ds.processed_xmls)} files")
                for xml in ds.processed_xmls:
                    print(f"      - {xml.name}")
            
            print()
        
        print("\n" + "="*80)
        print("KEY FEATURES:")
        print("="*80)
        print("""
1. ✅ XMLs are now GROUPED by suffix:
   - dataset.xml_groups['base'] → XMLs for base TIF
   - dataset.xml_groups['processed'] → XMLs for processed versions
   - Other suffixes automatically detected
   
2. ✅ Convenient properties:
   - dataset.base_xmls → Quick access to base XMLs
   - dataset.processed_xmls → Quick access to processed XMLs
   
3. ✅ Backward compatible:
   - dataset.xml_paths → Still contains ALL XMLs (flat list)
   
4. ✅ Separate but referenced:
   - Each base file can have multiple XML variants
   - All are mapped to the same dataset
   - Can be accessed separately when needed
   
5. ✅ Flexible name matching:
   - '50 nm_4' matches '50nm_4.xml' (flexible matching)
   - Automatically detects 'processed' variants
        """)
        
        # Show summary statistics
        print("\n" + "="*80)
        print("SUMMARY STATISTICS:")
        print("="*80)
        
        total_base = sum(len(ds.base_xmls) for ds in datasets)
        total_processed = sum(len(ds.processed_xmls) for ds in datasets)
        total_other = sum(len(xmls) for ds in datasets for suffix, xmls in ds.xml_groups.items() 
                         if suffix not in ['base', 'processed'])
        
        print(f"Total datasets: {len(datasets)}")
        print(f"Base XMLs: {total_base}")
        print(f"Processed XMLs: {total_processed}")
        print(f"Other XMLs: {total_other}")
        print(f"Total XMLs: {total_base + total_processed + total_other}")
        
    else:
        print("No datasets found.")
else:
    print(f"Test folder not found: {test_folder}")

print("\n" + "="*80)
print("USAGE EXAMPLE:")
print("="*80)
print("""
from hydro_analysis.core import find_dataset_files

# Load datasets
datasets = find_dataset_files(Path("Data/experiment"))

for ds in datasets:
    print(f"Processing: {ds.base_name}")
    
    # Process base XMLs
    for xml in ds.base_xmls:
        # These are XMLs for the original TIF
        tracks = TrackLoader.from_trackmate_xml(xml)
        analyze_original(tracks)
    
    # Process processed XMLs separately
    for xml in ds.processed_xmls:
        # These are XMLs for processed versions
        tracks = TrackLoader.from_trackmate_xml(xml)
        analyze_processed(tracks)
    
    # Access specific groups
    if 'base' in ds.xml_groups:
        print(f"Found {len(ds.xml_groups['base'])} base XMLs")
    
    # Check all available variants
    for suffix in ds.xml_groups.keys():
        print(f"Suffix '{suffix}': {len(ds.xml_groups[suffix])} XMLs")
""")
