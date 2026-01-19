"""Test script demonstrating the new separate datasets system.

WICHTIG: Prozessierte TIFs werden jetzt als SEPARATE Datasets behandelt!

Für die Beispiel-Struktur:
- 50 nm_4.tif          → Dataset 1 (mit 50 nm_4_Tracks.xml, 50nm_4.xml)
- 50 nm_4_processed.tif → Dataset 2 (mit 50 nm_4_processed_Tracks.xml, 50nm_4_processed.xml)

Beide nutzen die gleiche REC-Datei (50 nm_4.rec), aber haben separate XMLs!
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from hydro_analysis.core.io import DatasetIndex

print("="*80)
print("TEST: Separate Datasets für Original und Processed Files")
print("="*80)

test_folder = Path("Data")

if test_folder.exists():
    idx = DatasetIndex.from_root(test_folder)
    
    if idx.datasets:
        print(f"\n✅ Gefunden: {len(idx.datasets)} separate Datasets\n")
        
        # Group by base name (without _processed)
        grouped = {}
        for base_name in idx.list_bases():
            ds = idx.get(base_name)
            # Remove _processed to find base
            base = base_name.replace('_processed', '').replace('processed_', '')
            if base not in grouped:
                grouped[base] = {'original': None, 'processed': []}
            
            if '_processed' in base_name.lower():
                grouped[base]['processed'].append(ds)
            else:
                grouped[base]['original'] = ds
        
        # Display grouped
        for base, group in sorted(grouped.items()):
            print(f"📦 Gruppe: {base}")
            print("─" * 60)
            
            # Original
            if group['original']:
                ds = group['original']
                print(f"  📄 ORIGINAL: {ds.base_tif.name}")
                print(f"     REC: {ds.rec_path.name if ds.rec_path else 'None'}")
                print(f"     XMLs ({len(ds.xml_paths)}):")
                for xml in ds.xml_paths:
                    print(f"        - {xml.name}")
            
            # Processed
            for ds in group['processed']:
                print(f"  🔧 PROCESSED: {ds.base_tif.name}")
                print(f"     Pfad: {ds.base_tif.parent.name}/{ds.base_tif.name}")
                print(f"     REC: {ds.rec_path.name} (shared with original)")
                print(f"     XMLs ({len(ds.xml_paths)}):")
                for xml in ds.xml_paths:
                    print(f"        - {xml.name}")
            
            print()
        
        print("\n" + "="*80)
        print("WICHTIGE ÄNDERUNGEN:")
        print("="*80)
        print("""
1. ✅ Prozessierte TIFs sind SEPARATE Datasets
   - 50 nm_4.tif ist Dataset #1
   - 50 nm_4_processed.tif ist Dataset #2
   
2. ✅ Beide nutzen die gleiche REC-Datei
   - Aufnahme-Parameter bleiben gleich
   
3. ✅ Jedes hat seine eigenen XMLs
   - Original: 50 nm_4_Tracks.xml, 50nm_4.xml
   - Processed: 50 nm_4_processed_Tracks.xml, 50nm_4_processed.xml
   
4. ✅ processed_tifs Liste ist immer leer
   - Kein "Eltern-Kind" Verhältnis mehr
   - Alle sind gleichwertige, separate Datasets
   
5. ✅ Sucht in vielen Ordnern
   - Tracks/, Analysis/, preprocess/, preprocessed/, processed/
   - Inklusive Unterordner (preprocess/Tracks/, etc.)
        """)
    else:
        print("Keine Datasets gefunden.")
else:
    print(f"Test-Ordner nicht gefunden: {test_folder}")

print("\n" + "="*80)
print("ERWARTETE STRUKTUR (Beispiel):")
print("="*80)
print("""
Root/
  ├── 50 nm_4.tif              ← Dataset 1
  ├── 50 nm_4.rec              ← Shared REC
  ├── Tracks/
  │   ├── 50 nm_4_Tracks.xml           → Dataset 1
  │   └── 50 nm_4_processed_Tracks.xml → Dataset 2
  ├── Analysis/
  │   ├── 50nm_4.xml                   → Dataset 1 (matches with flexible naming!)
  │   └── 50nm_4_processed.xml         → Dataset 2
  └── preprocess/
      └── 50 nm_4_processed.tif  ← Dataset 2

ERGEBNIS: 2 separate Datasets!
  Dataset 1 = 50 nm_4.tif mit seinen XMLs
  Dataset 2 = 50 nm_4_processed.tif mit seinen XMLs
""")
