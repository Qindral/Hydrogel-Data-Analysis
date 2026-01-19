# Grouped XML Mapping - Update

## Neue Features (18. Januar 2026)

### Problem
Bisher wurden alle XMLs für eine TIF-Datei in einer flachen Liste gespeichert. Es war nicht einfach zu unterscheiden zwischen:
- XMLs für die Original-Datei (`50 nm_4.tif`)
- XMLs für prozessierte Versionen (`50 nm_4_processed.tif`)

### Lösung
XMLs werden jetzt **nach Suffix gruppiert** und bleiben **separat referenzierbar**.

## DatasetFiles Struktur (Erweitert)

### Alte Attribute (behalten für Kompatibilität)
```python
dataset.xml_paths: List[Path]  # Alle XMLs (flache Liste)
```

### Neue Attribute
```python
dataset.xml_groups: Dict[str, List[Path]]
# Dictionary mit Gruppen:
# - 'base': XMLs für die Original-Datei
# - 'processed': XMLs für prozessierte Versionen
# - andere Suffixe werden automatisch erkannt

dataset.base_xmls: List[Path]      # Property → xml_groups['base']
dataset.processed_xmls: List[Path] # Property → xml_groups['processed']
```

## Beispiel: Wie XMLs gruppiert werden

### Dateistruktur
```
Root/
  ├── 50 nm_4.tif
  ├── 50 nm_4.rec
  ├── Tracks/
  │   ├── 50 nm_4_Tracks.xml
  │   └── 50 nm_4_processed_Tracks.xml
  └── Analysis/
      ├── 50nm_4.xml
      └── 50nm_4_processed.xml
```

### Ergebnis
```python
dataset = datasets[0]

# Alte API (noch funktionsfähig)
dataset.xml_paths
# → [50 nm_4_Tracks.xml, 50nm_4.xml, 50 nm_4_processed_Tracks.xml, 50nm_4_processed.xml]

# Neue API (gruppiert)
dataset.xml_groups
# → {
#     'base': [50 nm_4_Tracks.xml, 50nm_4.xml],
#     'processed': [50 nm_4_processed_Tracks.xml, 50nm_4_processed.xml]
#   }

# Bequemer Zugriff
dataset.base_xmls
# → [50 nm_4_Tracks.xml, 50nm_4.xml]

dataset.processed_xmls
# → [50 nm_4_processed_Tracks.xml, 50nm_4_processed.xml]
```

## Verwendung

### Separate Verarbeitung von Base und Processed XMLs

```python
from hydro_analysis.core import find_dataset_files, TrackLoader

datasets = find_dataset_files(Path("Data/experiment"))

for dataset in datasets:
    print(f"Dataset: {dataset.base_name}")
    
    # Verarbeite Original-XMLs
    print(f"  Base XMLs: {len(dataset.base_xmls)}")
    for xml in dataset.base_xmls:
        tracks = TrackLoader.from_trackmate_xml(xml)
        # ... Analyse für Original-Daten
    
    # Verarbeite Processed-XMLs SEPARAT
    print(f"  Processed XMLs: {len(dataset.processed_xmls)}")
    for xml in dataset.processed_xmls:
        tracks = TrackLoader.from_trackmate_xml(xml)
        # ... Analyse für prozessierte Daten
```

### Zugriff auf alle Varianten

```python
for dataset in datasets:
    # Iteriere über alle Suffix-Gruppen
    for suffix, xmls in dataset.xml_groups.items():
        print(f"Suffix '{suffix}': {len(xmls)} XMLs")
        for xml in xmls:
            print(f"  - {xml.name}")
```

### Vergleich: Base vs. Processed

```python
for dataset in datasets:
    if dataset.base_xmls and dataset.processed_xmls:
        # Vergleiche Tracking-Ergebnisse
        base_tracks = TrackLoader.from_trackmate_xml(dataset.base_xmls[0])
        proc_tracks = TrackLoader.from_trackmate_xml(dataset.processed_xmls[0])
        
        print(f"Base: {base_tracks['particle'].nunique()} particles")
        print(f"Processed: {proc_tracks['particle'].nunique()} particles")
        print(f"Difference: {proc_tracks['particle'].nunique() - base_tracks['particle'].nunique()}")
```

## Gruppierungs-Logik

Die Funktion `_group_xmls_by_suffix()` verwendet folgende Regeln:

1. **Exakte Übereinstimmung** → `'base'`
   - `50 nm_4_Tracks.xml` bei base_name=`'50 nm_4'`
   - `50nm_4.xml` (mit flexibler Namensanpassung)

2. **Enthält 'processed'** → `'processed'`
   - `50 nm_4_processed_Tracks.xml`
   - `50nm_4_processed.xml`

3. **Zusätzliches Suffix** → Suffix als Schlüssel
   - `50 nm_4_variant1.xml` → `'variant1'`

4. **Keine Übereinstimmung** → `'other'`

## Vorteile

✅ **Separate Referenzierung**: XMLs für verschiedene Versionen sind getrennt zugänglich
✅ **Nicht kombiniert**: Base und processed werden nicht vermischt
✅ **Flexibel**: Automatische Erkennung von zusätzlichen Suffixen
✅ **Abwärtskompatibel**: `xml_paths` funktioniert weiterhin
✅ **Typsicher**: Properties bieten einfachen Zugriff

## Migration von altem Code

### Vorher
```python
for xml in dataset.xml_paths:
    # Problem: Base und Processed gemischt!
    tracks = TrackLoader.from_trackmate_xml(xml)
    process(tracks)
```

### Nachher (Option 1: Separate Verarbeitung)
```python
# Nur Base
for xml in dataset.base_xmls:
    tracks = TrackLoader.from_trackmate_xml(xml)
    process_base(tracks)

# Nur Processed
for xml in dataset.processed_xmls:
    tracks = TrackLoader.from_trackmate_xml(xml)
    process_processed(tracks)
```

### Nachher (Option 2: Mit Suffix-Info)
```python
for suffix, xmls in dataset.xml_groups.items():
    for xml in xmls:
        tracks = TrackLoader.from_trackmate_xml(xml)
        process(tracks, variant=suffix)
```

## Tests

Führen Sie den Test aus:
```bash
python hydro_analysis/test_grouped_xmls.py
```

Erwartete Ausgabe:
- Liste aller Datasets
- Gruppierung nach Suffix für jedes Dataset
- Statistiken über Base/Processed/Other XMLs

## Zusammenfassung

Die neue `xml_groups` Struktur ermöglicht es:
1. XMLs für verschiedene TIF-Varianten **separat zu referenzieren**
2. **Nicht zu kombinieren** (jede Gruppe bleibt getrennt)
3. Später **gezielt zuzugreifen** (z.B. nur Base oder nur Processed)
4. **Flexibel zu erweitern** (automatische Erkennung neuer Suffixe)

Alle XMLs werden gefunden und zugeordnet, bleiben aber logisch getrennt! ✅
