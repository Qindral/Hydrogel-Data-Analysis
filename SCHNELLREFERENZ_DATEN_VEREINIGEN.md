# Schnellreferenz: Daten Vereinigen

## Die 4 wichtigsten Funktionen

### 1️⃣ Ein Dataset laden (mehrere XMLs → ein DataFrame)

```python
from hydro_analysis.core import find_dataset_files, load_all_tracks_from_dataset

# Datasets finden
datasets = find_dataset_files(Path("Data/experiment"))

# Erstes Dataset laden (alle XMLs vereinigt)
tracks = load_all_tracks_from_dataset(
    dataset=datasets[0],
    min_length=30,
    combine_xmls=True
)
```

**Ausgabe**: Ein DataFrame mit allen Tracks aus allen XMLs des Datasets

---

### 2️⃣ Alle Datasets aus Ordner laden

```python
from hydro_analysis.core import load_all_datasets_from_folder

# Alle Datasets laden
datasets_dict = load_all_datasets_from_folder(
    root_path=Path("Data/experiment"),
    min_length=30
)

# Dictionary: name → tracks
for name, tracks in datasets_dict.items():
    print(f"{name}: {len(tracks)} Detections")
```

**Ausgabe**: Dictionary mit einem Eintrag pro Dataset

---

### 3️⃣ ALLES vereinigen (ein großer DataFrame)

```python
from hydro_analysis.core import load_and_combine_all_datasets

# ALLE Daten in einem DataFrame
all_tracks = load_and_combine_all_datasets(
    root_path=Path("Data/experiment"),
    min_length=30,
    add_dataset_column=True  # Spalte mit Dataset-Namen
)

print(f"{all_tracks['particle'].nunique()} Partikel total")
```

**Ausgabe**: Ein einzelner DataFrame mit ALLEN Tracks aus ALLEN Datasets

---

### 4️⃣ Nach Partikelgröße gruppieren

```python
from hydro_analysis.core import group_datasets_by_particle_size

# Automatisch nach Ordnernamen gruppieren
size_groups = group_datasets_by_particle_size(
    root_path=Path("Data/experiment"),
    min_length=30
)

# Dictionary: Größe_nm → vereinigte Tracks
for size_nm, tracks in sorted(size_groups.items()):
    print(f"{size_nm} nm: {tracks['particle'].nunique()} Partikel")
```

**Ausgabe**: Dictionary mit einem Eintrag pro Partikelgröße

---

## Vergleich der Funktionen

| Funktion | Eingabe | Ausgabe | Wann verwenden? |
|----------|---------|---------|-----------------|
| `load_all_tracks_from_dataset()` | 1 Dataset | 1 DataFrame | Einzelnes Experiment analysieren |
| `load_all_datasets_from_folder()` | Ordner | Dict[name → tracks] | Datasets separat verarbeiten |
| `load_and_combine_all_datasets()` | Ordner | 1 großer DataFrame | Alle Daten zusammen analysieren |
| `group_datasets_by_particle_size()` | Ordner | Dict[size → tracks] | Größenabhängige Analyse |

---

## Vollständiges Beispiel: MSD-Analyse

```python
from pathlib import Path
import trackpy as tp
from hydro_analysis.core import group_datasets_by_particle_size

# 1. Daten nach Größe gruppiert laden
size_groups = group_datasets_by_particle_size(
    root_path=Path("E:/PhD Data Analysis/SPT 2025 II/D_0 Wassermessung"),
    min_length=30
)

# 2. MSD für jede Größe berechnen
results = []
for size_nm, tracks in sorted(size_groups.items()):
    # MSD berechnen (trackpy macht das automatisch richtig!)
    msd = tp.imsd(
        tracks,
        mpp=tracks.attrs['mpp'],  # ← automatisch extrahiert!
        fps=tracks.attrs['fps']   # ← automatisch extrahiert!
    )
    
    # Diffusionskoeffizient aus MSD
    # MSD(t) = 4 * D * t  → D = MSD(t) / (4 * t)
    em = msd.mean(axis=1)
    D = em.iloc[1] / 4.0  # Erste Lagtime
    
    results.append({
        'size_nm': size_nm,
        'n_particles': tracks['particle'].nunique(),
        'D': D
    })
    
    print(f"{size_nm} nm: D = {D:.4f} µm²/s")

# 3. Ergebnisse speichern
import pandas as pd
df = pd.DataFrame(results)
df.to_csv('results.csv', index=False)
```

**Das war's!** Nur ~25 Zeilen Code für eine komplette Analyse!

---

## DataFrame-Struktur

Alle Funktionen geben DataFrames mit dieser Struktur zurück:

### Spalten
```python
['particle', 'frame', 'x', 'y']
```

Optional (bei `load_and_combine_all_datasets`):
```python
['particle', 'frame', 'x', 'y', 'dataset']
```

### Attribute (Metadaten)
```python
tracks.attrs['mpp']           # µm/px (automatisch extrahiert!)
tracks.attrs['fps']           # Frames/Sekunde
tracks.attrs['mode']          # "60 FPS" oder "20 FPS"
tracks.attrs['dataset_name']  # Name des Datasets
```

---

## Wichtige Parameter

### `min_length`
Minimale Track-Länge (Anzahl Frames)
- **Standard**: 10
- **Empfohlen**: 30 für MSD-Analyse
- **Effekt**: Tracks mit < min_length werden entfernt

### `combine_xmls`
Sollen mehrere XMLs eines Datasets vereinigt werden?
- **True**: Ein DataFrame pro Dataset (empfohlen)
- **False**: Liste von DataFrames (ein pro XML)

### `add_dataset_column`
Soll Spalte mit Dataset-Namen hinzugefügt werden?
- **True**: Spalte 'dataset' wird hinzugefügt (empfohlen für Gruppierung)
- **False**: Keine zusätzliche Spalte

---

## Troubleshooting

### Problem: "No datasets found"
**Lösung**: Prüfen Sie, ob `.rec` Dateien vorhanden sind. Das System benötigt REC-Dateien für jedes Dataset.

### Problem: "No XML files found"
**Lösung**: XMLs müssen den gleichen Basisnamen wie die TIF-Datei haben (ohne Präfixe/Suffixe).

### Problem: "Could not extract particle size"
**Lösung**: Ordnernamen müssen Partikelgröße enthalten: `"50nm"`, `"100_nm"`, `"200 nm"` etc.

---

## Logging aktivieren

```python
import logging

# INFO Level (empfohlen)
logging.basicConfig(level=logging.INFO)

# DEBUG Level (alle Details)
logging.basicConfig(level=logging.DEBUG)

# WARNING Level (nur Warnungen/Fehler)
logging.basicConfig(level=logging.WARNING)
```

---

## Weiterführende Dokumentation

- **Vollständige Anleitung**: [UNIFIED_LOADING_GUIDE.md](UNIFIED_LOADING_GUIDE.md)
- **Test-Suite**: [test_unified_loading.py](hydro_analysis/test_unified_loading.py)
- **Praxis-Beispiel**: [example_unified_loading_analysis.py](hydro_analysis/example_unified_loading_analysis.py)
- **Technische Details**: [CORE_IO_REPAIR_SUMMARY.md](CORE_IO_REPAIR_SUMMARY.md)

---

## Zusammenfassung

**Eine Zeile Code** → Alle Daten laden und vereinigen!

```python
all_tracks = load_and_combine_all_datasets(Path("Data/experiment"), min_length=30)
```

**Viel Erfolg! 🚀**
