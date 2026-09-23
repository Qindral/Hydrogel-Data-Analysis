# Validierung: aktive Auswertungen

## SNR und DoG-Detektionsrobustheit

Für diese Aufgabe gibt es **zwei aktive Einstiege**:

| Skript | Eingaben | Aufgabe und Ausgaben |
|---|---|---|
| `SNR_DoG_Compute.py` | vollständige XML-Sessions, TIFF, `.rec` | Inventar, SNR, Ausschlussprotokoll, echte DoG-Parameterprüfung; CSV und `cache/snr_dog.pkl` |
| `SNR_DoG_Figures.py` | ausschließlich dieser neue Cache | Trennungsverteilungen, Beispieltafel, Wiederfindungsrate; PNG, 600 dpi |

Ausgaben liegen standardmäßig in `analysis_results/snr_dog` im Projekt.
So überschreibt diese Konsolidierung keine älteren Dissertationsabbildungen.
`--output` erlaubt den gewünschten endgültigen Exportordner.

```powershell
python -m hydro_analysis.MSD_Trackmate.Validation_Claude.SNR_DoG_Compute --inventory-only
python -m hydro_analysis.MSD_Trackmate.Validation_Claude.SNR_DoG_Compute
python -m hydro_analysis.MSD_Trackmate.Validation_Claude.SNR_DoG_Figures
```

`--no-sensitivity` überspringt ausschließlich die Fiji-Parameterprüfung.
`--figures separation examples` exportiert gezielt diese zwei Diagrammtypen.
Es werden keine GUI-Fenster geöffnet. Ein Manifest listet die erzeugten PNGs.
Vorhandene ältere PNGs werden nicht automatisch gelöscht; maßgeblich ist das
Manifest des aktuellen Laufs. Es gibt keine 3D-Oberflächen, KDE-Duplikate oder
zusätzlichen annotierten Varianten mehr.

### Eingaberegeln

- Wasser, Hydrogel und immobilisierte Partikel bleiben getrennt.
- Innerhalb einer Bedingung werden Partikelgröße und mpp getrennt dargestellt.
- Nur `DOG_DETECTOR` wird verwendet; kein Standardradius und kein LoG-Fallback.
- Exakte Zuordnung über den TIFF-Dateinamen in der XML. Neuere `*NEW*`-Ordner
  haben Vorrang. Mehrdeutige Zuordnungen werden ausgeschlossen statt geraten.
- Positionen, Radius, Kanal und Vorfilterung stammen aus derselben Session.
- `AllSpots` wird zur Hintergrund-/Nachbarausschlussmaske verwendet. Für
  Partikelmessungen gelten Sichtbarkeit und die gespeicherten Spot-Filter;
  Track-Filter oder Mindesttracklängen werden nicht zur SNR-Selektion verwendet.
- Aktuell unterstützt: 2D-Einkanal-TIFFs, quadratische Pixel. Andere Formate
  werden in `failures.csv` gemeldet.

### Methodik und Grenzen

Zufällig bis zu 24 Frames pro Movie, darin bis zu 12 isolierte Detektionen;
Seed 42 plus stabiler Datei-Hash. Unscharfe Frames werden mit dem vorhandenen
Varianz-des-Laplacians/MAD-Verfahren ausgeschlossen. Alle XML-Spots blockieren
Nachbarschaft und Hintergrund, auch nicht sichtbare oder ungetrackte Spots.
Detektionen mit Abstand unter `1.5*(Radius_i+Radius_j)` werden ausgeschlossen.
Zusätzlich werden breite Intensitätsprofile oberhalb Median + max(3.5 robuste
Standardabweichungen, 20% des Medians), innerhalb eines Movies, ausgeschlossen.
Das ist ein nachvollziehbarer Aggregatverdachtsfilter, kein sicherer Nachweis
einzelner Partikel; nicht aufgelöste Aggregate können unerkannt bleiben.

Für jeden Frame gibt es zufällige **Referenz-Hintergrundpatches** und davon
getrennte **Kontrollpatches**. Sie überlappen sich nicht und halten Abstand zu
allen detektierten Spots. Keine Auswahl anhand besonders niedriger Intensität.
Partikel und Kontrollpatches verwenden dieselbe Peak-Suchfläche und dieselben
Referenzwerte: `SNR = (Peak - Referenzmittelwert) / Referenzstandardabweichung`.
Zu dicht besetzte Frames ohne ausreichenden sauberen Hintergrund werden
protokolliert und ausgeschlossen; es gibt keinen stillen lokalen Ersatz.

Die Analyse-DoG nutzt die Skalenkonvention aus TrackMates `DogDetector` und
ImgLib2 `DifferenceOfGaussian.computeSigmas`: 0.9/1.1 um Radius/sqrt(2),
mit Korrektur für die angenommene Eingangsunschärfe. SciPy und ImgLib2 sind
numerisch nicht als identisch garantiert. Die Analyse-DoG-SNR wird deshalb
**nicht** mit XML-`QUALITY` gleichgesetzt und nicht gegen XML-`THRESHOLD` getestet.
Quellen: [DogDetector](https://github.com/trackmate-sc/TrackMate/blob/master/src/main/java/fiji/plugin/trackmate/detection/DogDetector.java),
[computeSigmas](https://github.com/imglib/imglib2-algorithm/blob/master/src/main/java/net/imglib2/algorithm/dog/DifferenceOfGaussian.java).

Die Wiederfindungsprüfung führt dagegen den **echten Fiji-DoG-Detektor** aus:
3x3 Kombinationen aus 80/100/120% des XML-Radius und XML-Schwellenwerts auf
denselben ausgewählten Frames. Matching-Toleranz max(1 Pixel, Radius/2),
eindeutige Zuordnung. Referenz sind die verbliebenen ursprünglichen Detektionen;
es findet kein Tracking und keine nachträgliche XML-Spotfilterung der neuen
Detektionen statt. Auch die Wiederfindung bei 100/100% wird ausgewiesen.
Die Heatmaps mitteln Wiederfindungsanteile mit gleichem Gewicht pro Movie.

Die ECDFs gewichten ebenfalls jedes Movie gleich. Einzelbilder sind keine
unabhängigen biologischen Replikate. Der berichtete Anteil über dem 99%-Quantil
der Hintergrundkontrollen ist eine deskriptive Trennungskennzahl, keine
Detektionsgenauigkeit, Sensitivität oder Falschpositivrate mit Ground Truth.
Die Aussage gilt für **scharfe, isolierte und bereits detektierte** Partikel.

### Dateien lesen

| Datei | Bedeutung |
|---|---|
| `session_inventory.csv` | alle gefundenen Session-Kandidaten, Priorität/Verwendung |
| `log_sessions.csv` | alle gefundenen LoG-Sessions, einschließlich bereits ersetzter Versionen |
| `files_needing_dog.csv` | nachzuliefernde DoG-Sessions oder zu klärende Zuordnungen/Kalibrierungen |
| `snr_measurements.csv` | Partikel und Hintergrundkontrollen, DoG-SNR und Rohbild-SNR |
| `exclusions.csv` | ausgeschlossene Frames/Detektionen und Gründe |
| `failures.csv` | nicht auswertbare Dateien bzw. fehlgeschlagene Fiji-Läufe |
| `sensitivity.csv` | Wiederfindungszähler je Movie und Parameterkombination |
| `snr_summary_by_movie.csv` | Trennungskennzahlen je Movie; Eingang von `Validation_Summary.py` |
| `example_selection.csv` | Herkunft der gezeigten Beispiele |
| `figure_manifest.json` | vollständige Liste der aktuellen Diagramme |

Beispiele werden innerhalb einer Bedingung aus der Gruppe mit niedrigster
medianer Partikel-DoG-SNR ausgewählt. Schwaches Beispiel nahe dem 15%-Quantil;
stärkeres Beispiel und medianer Hintergrund aus **demselben Movie und Frame**.
Einheitliche Farbskalen je Bildzeile. Ohne passendes Tripel keine Beispieltafel.

### Bereinigte ältere Einstiege

`TaskC_Detectability_Compute.py` leitet nur noch zur neuen Berechnung weiter.
`TaskC_Detectability_Figure.py` und `TaskC_Detectability_Distribution.py` leiten
zur einen Abbildungsausgabe weiter. Diese kleinen Kompatibilitätseinstiege
verhindern kaputte Verweise, enthalten aber keine eigene Analyse mehr.
`Detectability_Sweep_Demo.py` erzeugt keine alten Diagnosebilder mehr.
Die doppelte Beispielauswahl `_detectability_selection.py` wurde entfernt.
Der SNR-Teil von `Validation_GPT/complete_spt_validation.py` nutzt dieselbe
Berechnung und dieselben Abbildungen, mit dem dort übergebenen Manifest.
`detectability_all_sizes.pkl` wird für diese Aufgabe nicht mehr gelesen.

## Weitere Aufgaben — nicht Teil dieser SNR-Umstellung

| Bereich | Einstieg / Zweck |
|---|---|
| Lokalisierungsfehler | `Loc_Error_Analyse_immob_particle.py`: bereinigte Positionsstreuung, laedt die Referenz-Schrittweite aus `iMSD_histograms.py`'s Cache. `IMMOB_DATASETS`/`_corrected_particle_size()` leben hier jetzt zentral (auch von `PSF_Analysis.py` importiert) |
| PSF | `PSF_Analysis.py`: Gaußfits einzelner Frames (max. 200 Tracks/Gruppe x 20 Frames, gecacht), asymmetrischer Violinplot PSF width vs. fitting pixel diameter -- ersetzt `PSF_Analysis_Immobilized.py` |
| MSD-Korrelationen/-Korrektur | `Correlations.py`: pro-Datei loc_err-Korrektur (Endresultate, eMSD), Tracklängen-Bin-Vergleich, D-vs-n, Qualitäts-Korrelationen -- ersetzt `TaskA2_A4_Intercept_Comparison.py`, `TaskA3_MSD_Correction.py`, `TaskB2_LengthBias_Scatter.py`, `MSD_Diffusion_Correlations_Test_D0.py`, `MSD_Diffusion_Correlations_Test_20mg.py` und `Dissertation_Figures/DiffusionExponent_Correlation.py` (kein TrackMate-Settings-Sweep, keine Intercept-Diagnostik mehr) |
| Tracklängen-Bias | `TaskB1_TrackLengthBias_Compute.py` (Cache, bleibt bestehen -- wird von `iMSD_histograms.py` gebraucht). `TaskB2_LengthBias_Scatter.py`/`TaskB3_MinTrackLength_Sensitivity.py` nach `obsolete/` verschoben, ersetzt durch `Correlations.py`'s Tracklängen-Bin-Vergleich |
| Pro-Track/-File-Verteilungen | `iMSD_histograms.py` (pro Track), `eMSD_files_histogram.py` (pro File): Tracklänge/Schrittweite/D/n je Partikelgröße, Wasser vs. Hydrogel |
| Fit-Punkte-Robustheit | `Fittingpoints_robustness.py`: D/n-Stabilität vs. Anzahl MSD-Fit-Punkte (4-8) |
| SNR / Detektierbarkeit | `SNR_Analysis.py`: Peak/Hintergrund-SNR-Übersicht je Partikelgröße und Bedingung, baut auf `snr_signal_profile.py` auf |
| Infrastruktur | XML-Reader in `Validation_GPT/trackmate_session.py`, `_trackmate_settings_parser.py`, `Compare_Tracks_vs_RawAnalysis.py`, `Frame_Sharpness_Immobilized.py`. Fiji-Headless-Runner nach `obsolete/` verschoben (Headless-TrackMate-Vergleich überholt) |

Diese Aufgaben wurden nicht pauschal gelöscht oder in die SNR-Berechnung
eingemischt. Details stehen in ihren jeweiligen Modulbeschreibungen.
