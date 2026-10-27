# Entscheidungen

Hier steht jede Entscheidung über Daten, Methode oder Deutung, die nicht die KI treffen darf.
Offene Einträge blockieren die betroffene Auswertung. Entschiedene Einträge bleiben stehen und
sind der Beleg für den Methodenteil.

Format (wird vom Rainmeter-Skin gelesen, bitte beibehalten):
- Überschrift: `## E000 · OFFEN · JJJJ-MM-TT` oder `## E000 · ENTSCHIEDEN · JJJJ-MM-TT`
  (bei OFFEN das Datum, an dem die Frage aufkam; bei ENTSCHIEDEN das Datum der Entscheidung)
- direkt darunter eine Zeile `**Frage:** …`
- zum Entscheiden: Status ändern, Datum setzen, die Zeile `**Entscheidung:**` ausfüllen,
  danach die Zeile im Register in `AGENTS.md` und `CLAUDE.md` (Abschnitt 0.2) anpassen.

## E001 · GEschlossen · 2026-10-06
**Frage:** Randfilter behalten? Detektionen in den äußeren 3 % des Bildes werden gelöscht. 
**Stelle:** `core.io.remove_edge_artifacts`, läuft automatisch in `single_file_data`; im Code seit 2026-07-07.
**Wirkung heute:** entfernt in Wasser 0,5 bis 9,6 % der Detektionen je Partikelgröße, in einzelnen Dateien über 50 %; Tracks werden an der Lücke geteilt.
- A: behalten (3 %) → im Methodenteil begründen, Zahl der entfernten Detektionen je Bedingung berichten.
- B: abschalten → mehr Detektionen und längere Tracks; mögliche Fehlverknüpfungen am Rand bleiben in den Daten.
- C: Sensitivität rechnen (0, 1, 3 %) und danach festlegen → ein zusätzlicher Durchlauf, zeigt, ob D davon abhängt.
**Entscheidung:** A Rand ist immer von unsicherheiten tracking linking betroffen und muss ausgeschlossen werden.
muss nicht weiter begründet werden. 

## E002 · ENTSCHIEDEN · 2026-10-27
**Frage:** Womit werden die Dateien beim Mittelwert je Partikelgröße gewichtet?
**Stelle:** `core.analysis.weighted_average_per_size`, Gewicht = `num_tracks`.
**Wirkung heute:** `num_tracks` wird vor dem 10-Frame-Filter gezählt. Bei 50 nm im Hydrogel sind 66 % dieser Tracks zu kurz und gehen nicht in den Fit ein, bestimmen aber das Gewicht.
- A: Trackzahl nach dem 10-Frame-Filter → Gewicht entspricht den Tracks, die wirklich im Fit stecken.
- B: ungewichtet, jede Datei zählt gleich → passt zur Regel „das Movie ist die Einheit“; Dateien mit wenigen Tracks wiegen dann so viel wie große.
- C: so lassen → D und Fehlerbalken bleiben wie bisher, die Gewichtung ist dann zu begründen.
**Entscheidung:** A offen. Im Dialog bestätigt: A, Gewicht = Tracks nach dem 10-Frame-Filter (`num_tracks_fit`) (Nutzer, 2026-10-27). Umgesetzt in `MSD_FromTrackmate_20mg.py` (`WEIGHT_KEY`): 35 nm 1183, 50 nm 539 Tracks.

## E003 · ENTSCHIEDEN · 2026-10-27
**Frage:** Mindest-Bildrate von 40 Hz für nominal 20, 50 und 100 nm behalten?
**Stelle:** `MSD_FromTrackmate_D0.py` und `MSD_FromTrackmate_20mg.py`, `MIN_FPS_SMALL_SIZES`.
**Wirkung heute:** unbekannt. Übersprungene Dateien werden nur in der Konsole genannt und nirgends gespeichert. Zuerst zählen lassen, welche Dateien betroffen sind.
- A: behalten → übersprungene Dateien in die Ausschlussliste schreiben und im Methodenteil nennen.
- B: abschaffen → alle Dateien gehen ein; langsam aufgenommene Filme kleiner Partikel haben wenige, große Schritte.
- C: andere Schwelle → mit Begründung über Schrittweite je Frame festlegen.
**Entscheidung:** A Ja. langsamere fps ausschließen (Nutzer, 2026-10-27). Hydrogel 20 mg/mL: 2 Dateien übersprungen und in `msd_20mg_exclusions.csv` vermerkt (`20nm_1d_B4_01_processed` 33,3 Hz, `B3_inside_1d_1mmfrom injection_50_20mg` 20 Hz). Wasser (E040 A): keine Datei unter 40 Hz (`20 nm_water` mit genau 40,0 Hz bleibt).

## E004 · ENTSCHIEDEN · 2026-10-27
**Frage:** Wie viele Lag-Punkte gehen in den MSD-Fit: 6 (Standard im Code) oder 4 (in den Validierungsläufen benutzt)?
**Stelle:** `core.analysis.DEFAULT_MSD_FIT_POINTS = 6`.
**Wirkung heute:** D und α im Kapitel stammen aus 6 Punkten, die Validierung zum Tracklängen-Bias aus 4.
- A: 6 Punkte überall → Validierung mit 6 wiederholen.
- B: 4 Punkte überall → Hauptauswertung neu rechnen.
- C: Sensitivität 2 bis 6 Punkte einmal berichten und einen Wert festlegen → beantwortet die Frage auch für die Prüfer.
**Entscheidung:** A 6 Lagpunkte (Nutzer, 2026-10-27). Validierung mit 6 Punkten noch nicht wiederholt.

## E005 · ENTSCHIEDEN · 2026-10-27
**Frage:** Wie wird der Lokalisationsfehler berichtet, wenn der Achsenabschnitt der MSD negativ ist?
**Stelle:** `core.analysis.perform_msd_analysis`, `sigma_loc_nm`; Mittelung in `weighted_mean_and_err`.
**Wirkung heute:** Ein negativer Achsenabschnitt wird zu „kein Wert“ und fällt aus dem Mittel. In Wasser betrifft das 22 von 44 Dateien, im Hydrogel 3 von 33. Der gemittelte Wert stammt also nur aus Dateien mit positivem Achsenabschnitt.
- A: σ aus den immobilisierten Partikeln als Hauptwert, Achsenabschnitt nur als Diagnose zeigen → kein Auswahleffekt.
- B: über die Achsenabschnitte aller Dateien mitteln (auch negative), erst danach die Wurzel ziehen → Mittel ohne Auswahl, kann nahe null oder negativ ausfallen.
- C: so lassen → im Text angeben, dass nur positive Achsenabschnitte eingehen und wie viele Dateien das sind.
**Entscheidung:** A (Nutzer, 2026-10-27). Hydrogel: `sigma_loc_nm` = σ der immobilisierten Partikel (E039), Achsenabschnitt als `sigma_loc_intercept_nm` nur Diagnose.

## E006 · ENTSCHIEDEN · 2026-10-27
**Frage:** Was passiert mit Dateien ohne Kalibrierung (.rec fehlt, Bildbreite unbekannt) oder mit fehlgeschlagenem Fit?
**Stelle:** `core.io.single_file_data` (gibt `None` zurück), `core.analysis.analyze_multiple_files` (fängt Fehler ab).
**Wirkung heute:** Die Datei wird übersprungen, es gibt nur eine Konsolenmeldung. In den gespeicherten Ergebnissen ist nicht zu sehen, ob und welche Dateien fehlen.
- A: Abbruch mit Fehlermeldung, die Datei und fehlendes Feld nennt → keine Datei kann unbemerkt fehlen; der Lauf stoppt, bis das Problem behoben ist.
- B: überspringen, aber in die Ausschlussliste schreiben → der Lauf geht durch, das Fehlen ist dokumentiert.
**Entscheidung:** A (Nutzer, 2026-10-27). Die .rec zu `B3_inside_1d_1mmfrom injection_50_20mg` heißt `B2_inside_1d_1mmfrom injection_50_20mg.rec` und ist im Skript zugeordnet (`REC_OVERRIDES`, Nutzer, 2026-10-27). Wasser: `1000_nm_10_Tracks_v2.xml` ist eine zweite Analyse des Films `1000_nm_10` und wird mit `1000_nm_10.tif.rec` kalibriert (Nutzer, 2026-10-27).

## E007 ·Geschlossen · 2026-10-06
**Frage:** DLS-Messungen mit Polydispersität über 42 % ausschließen? 
**Stelle:** `Litesizer/*`, `MAX_PDI = 42.0`.
**Wirkung heute:** Messungen über der Schwelle fehlen in Größen und D₀ aus der DLS; die Zahl wird nur in der Konsole genannt.
- A: behalten → Schwelle begründen, Zahl der ausgeschlossenen Messungen je Größe berichten.
- B: alle Messungen zeigen und die auffälligen markieren → kein Ausschluss, breitere Streuung.
- C: andere Schwelle → mit Quelle oder Herstellerangabe festlegen.
**Entscheidung:** A Ja ab eine Polydispersität ist das Ergebnis zu unsicher unter 0.3 sehr genau. 6-10-2026

## E008 · ENTSCHIEDEN · 2026-10-08
**Frage:** Rheologie: Läufe mit dem Kennzeichen `group_outlier` (Faktor 2 gegen den Median gleich alter Läufe) aus der gefilterten Statistik lassen?
**Stelle:** `Rheology/Rheo_Hyd20_Compute.py`, `STATS_EXCLUDE_QC_CODES`, `QC_GROUP_OUTLIER_FACTOR`.
**Wirkung heute:** Diese Läufe bleiben in Tabellen und Abbildungen, fehlen aber in der Statistik `qc_filtered`. Betroffen: Hyd20 (26) (13 d, G' = 2865 Pa gegen Median 1104 Pa).
- A: behalten → Kriterium im Methodenteil nennen.
- B: nur markieren, nicht ausschließen → Statistik über alle gültigen Läufe.
**Entscheidung:** A, Statistik mit `qc_filtered` wie bisher, zusammen mit den Codes axial_force, presheared und cracks (Nutzer, 2026-10-08, siehe E025).

## E009 · ENTSCHIEDEN · 2026-10-27
**Frage:** Driftkorrektur im MSD-Code: entfernen, behalten oder richtig einführen? Driftkorrektur nur bei Wasser.
**Stelle:** `core.analysis.perform_msd_analysis`, Bedingung „Partikel je Frame > 50“.
**Wirkung heute:** keine. Die Bedingung wurde bei keiner der 77 Dateien ausgelöst (höchster Wert 1,1), es wird also nie Drift abgezogen. Der Code steht aber da und würde bei dichteren Filmen still eingreifen.
- A: entfernen → der Code tut, was im Methodenteil steht: keine Driftkorrektur.
- B: behalten → als Regel dokumentieren, obwohl sie nie greift.
- C: Drift prüfen und dann entscheiden → einmal die mittlere Verschiebung je Film ausgeben lassen.
**Entscheidung:** B - ab 30 Partikeln pro Frame (Nutzer, 2026-10-27). Umgesetzt mit dem bisherigen Maß (Tracks / Frames ≥ 30, `DRIFT_MIN_TRACKS_PER_FRAME`); Hydrogel höchstens 0,25, keine Driftkorrektur.

## E010 · Geschlossen · 2026-10-06
**Frage:** Sollen die 40 neu gefundenen DLS-Größenmessungen in die DLS-Referenzwerte (Größe, D₀, Legenden-Labels) eingehen? 

**Stelle:** `Litesizer/litesizer_visualization.py`, `litesizer_measurements_mean.py`, `litesizer_size_distribution.py` lesen nur `Size_repitition_All Sizes.xlsx`; die erweiterte Tabelle schreibt `Litesizer/litesizer_measurement_inventory.py` (`dls_size_table_extended.csv`).
**Wirkung heute:** keine. Die Referenzwerte stammen aus den 30 Messungen vom 15. und 20.01.2026 (20 °C). Von den 40 neuen (PDI ≤ 42 %) sind 22 bei 25 °C, 11 ohne bekannte Temperatur (nur CSV), 7 bei 20 °C (alle 12.12.2025); 5 haben das Lösungsmittelmodell 50 mM TrisHCl; 7 haben keine Größe im Namen (u. a. Wasser-Leerprobe 08.10., „Untitled“).
- A: nicht einbeziehen → Referenzwerte bleiben wie bisher.
- B: nur die 7 Messungen bei 20 °C mit Größe im Namen (12.12.2025) einbeziehen → zusätzliche Wiederholungen für 20, 50, 100, 200 und 500 nm unter gleicher Temperatur.
- C: alle 40 einbeziehen → mehr Wiederholungen, aber gemischte Temperaturen, Lösungsmittelmodelle und Messmodi (5 × Quick).
**Entscheidung:** Nein nur als Anhang - es sind Messungen, die strenggenommen auszuschließen sind - aber insgesamt die Realibität der Messung unterstützen -> Anhang 6-10-2026

## E011 · ENTSCHIEDEN · 2026-10-07
**Frage:** Autofluoreszenz: Wie werden die Abklingkurven normiert, welche Kurven gehen ein und wie wird t½ bestimmt?
**Stelle:** `Auto_fluorecence/autofluorescence_analysis.py`, `normalise_switch_on`, `I0_OFFSET_FRAMES`, `HALF_LEVEL`.
**Wirkung heute:** 15 Einschaltkurven aus 12 Aufnahmen; t½ in 11 erreicht, in den 4 erneuten Einschaltungen nach einer Aus-Phase nicht innerhalb der Kurve. Je Schaltvorgang wird der teilbelichtete Übergangsframe nicht verwendet (19 Frames).
- Normierung: A: (I − I_dunkel)/(I₀ − I_dunkel), I₀ = Frame nach dem Übergangsframe → Kurve startet bei 1, 0 = Laser aus. B: wie A, I₀ = Maximum der ersten Frames. C: I/I₀ ohne Abzug → Kamera-Offset (~810 counts) bleibt, 0,5 wird oft nicht erreicht.
- t½: A: erster Frame mit I_norm ≤ 0,5, ohne Fit und Glättung. B: aus exponentiellem Fit. C: beides.
- Kurven: A: nur erstes Einschalten je Datei. B: alle Einschaltvorgänge.
- Legende: A: Wellenlänge + Leistung + Konzentration. B: Bestrahlungsstärke (W/cm²). C: nur Leistung.
**Entscheidung:** Normierung A, t½ A, Kurven B, Legende C (Nutzer, 2026-10-07). Legende am selben Tag durch E012 ersetzt.

## E012 · ENTSCHIEDEN · 2026-10-07
**Frage:** Autofluoreszenz: Was steht in der Legende, und wie werden Pause und Wiederanstieg nach dem Ausschalten angegeben?
**Stelle:** `Auto_fluorecence/autofluorescence_analysis.py`, `wavelength_nm`, `gel_concentration_mg_ml`, `pause_rise`, `WAVELENGTH_NM_NOT_IN_REC`, `PRE_PAUSE_OFFSET_FRAMES`.
**Wirkung heute:** 4 Gruppen in der Legende; 4 Pausen (4,9 bis 11,0 s) mit Wiederanstieg +1,1 bis +2,7 %. `20mg_ohnePartikel_B1.tif.rec` nennt keine Wellenlänge.
- Wellenlänge B1: A: 514 nm wie B1_02. B: 488 nm. C: „unbekannt“.
- Prozent: A: Anstieg gegenüber vor der Pause, dunkelabgezogen. B: zurückgekehrter Anteil der gebleichten Intensität. C: I₀,neu / I_vorher − 1 auf Rohcounts.
- I_vorher: A: Mittel der letzten 1 s. B: Einzelframe zwei Frames vor dem erkannten Ausschalten.
- Konzentration: A: 20/40/60 mg/mL aus dem Dateinamen. B: 20,8 mg/mL laut .rec für das 20-mg-Gel.
**Entscheidung:** Legende = Konzentration + Wellenlänge + Leistung; Wellenlänge B1 A (514 nm), Prozent C, I_vorher B, Konzentration A (Nutzer, 2026-10-07).

## E013 · ENTSCHIEDEN · 2026-10-07
**Frage:** Laserstrahl-Profile: durch welche Zeile/Spalte, in welcher Größe, und bleiben die Einzelbilder je Einstellung?
**Stelle:** `Laser_Intensity/laser_beam_characterization.py`, `centre_profiles`, `plot_profiles`.
**Wirkung heute:** 6 Einstellungen (488 nm: 1,065/2/2,56 mW; 514 nm: 0,6/1,2/1,8 mW) in einer Abbildung ohne und einer mit Gauß-Fit. Bei 514 nm und 514_2 liegt das Fit-Zentrum außerhalb des Bildes (x₀ ≈ −31/−33 µm); das y-Profil liegt dort in der Randspalte x = −29,85 µm.
- Zentrum: A: Fit-Zentrum, auf den Bildrand begrenzt. B: Bildmitte. C: hellstes Pixel.
- y-Achse: A: Bestrahlungsstärke E (W/m²). B: normiert auf das eigene Maximum.
- Einzelbilder: A: entfallen. B: bleiben.
**Entscheidung:** Zentrum A, y-Achse A, Einzelbilder A; Profile als einzelne Zeile/Spalte ohne Mittelung (Nutzer, 2026-10-07).

## E014 · OFFEN · 2026-10-07
**Frage:** Laserstrahl: Sollen die bestehenden Pixelkorrekturen (zeitliche Ausreißer, Hot Pixel) bleiben?
**Stelle:** `Laser_Intensity/laser_beam_characterization.py`, `load_measurement` (`TEMPORAL_OUTLIER_SIGMA`, `HOT_PIXEL_SIGMA`, `HOT_PIXEL_BOX_PX`).
**Wirkung heute:** je Datei 0 bis 11 Einzelwerte durch den zeitlichen Median ersetzt und 57 bis 153 Pixel des Mittelbildes (0,2 bis 0,4 %) durch den lokalen 5 × 5-Median; Zählung in `laser_beam_exclusions.csv`. Nicht im Register (Abschnitt 0.2).
- A: behalten → im Register eintragen, im Methodenteil nennen.
- B: entfernen → E und Fit ändern sich leicht; einzelne Übersteuerungs- oder Hot-Pixel-Werte gehen in Profile und Fit ein.
- C: nur Hot-Pixel-Korrektur behalten → die wenigen zeitlichen Ausreißer bleiben im Mittelbild.
**Entscheidung:** offen

## E015 · ENTSCHIEDEN · 2026-10-07
**Frage:** Laserstrahl, neue Version `Laser_beam_norm_detector.py`: Wie werden Leistung, Bildfläche, Korrekturen, Strahlgröße, beleuchtete Fläche, Fitmodell, E_max und Abbildungen festgelegt?
**Stelle:** `Laser_Intensity/Laser_beam_norm_detector.py`.
**Wirkung heute:** Der Strahl ist in keiner Messung vollständig im Bild (Anteil des Gauß-Fits im SPT-Fenster 16 bis 49 %); E ist daher eine Obergrenze. Die 1/e²-Ellipse überdeckt bei allen 6 Messungen das ganze Fenster, E_mean_beleuchtet = P/A_Fenster.
- Leistung: A: wie Vorgabe, P ganz auf das Bild verteilt. B: zusätzlich gaußkorrigiert. C: nur gaußkorrigiert (E sänke um 51 % bei 488 nm, 71 bis 84 % bei 514 nm).
- Bildfläche: A: SPT-Fenster 60 × 45 µm für alle. B: ganzes aufgenommenes Bild.
- Korrekturen: A: wie bisher. B: keine Pixelkorrekturen, nur Hintergrund (dunkelstes Plateau). C: auf E014 warten.
- Strahlgröße: A: Fit + D4σ markiert. B: nur Fit, kein D4σ. C: nur gemessen.
- Beleuchtete Fläche: A: Pixel ≥ 1/e² des Maximums. B: innerhalb der gefitteten 1/e²-Ellipse. C: ganzes Fenster.
- Fitmodell: A: achsenparallel. B: gedrehte Ellipse. C: beide.
- E_max: A: gemessen und Fit. B: nur Fit. C: nur gemessen.
- Abbildungen: nur die x/y-Linienprofile der Bestrahlungsstärke mit Fit.
**Entscheidung:** Leistung A, Bildfläche A, Korrekturen B, Strahlgröße B, beleuchtete Fläche B, Fitmodell B, E_max C, Abbildungen: nur Linienprofile (Nutzer, 2026-10-07).

## E016 · ENTSCHIEDEN · 2026-10-07
**Frage:** FRAP v2: Welcher hydrodynamische Radius gilt für das 150-kDa-FITC-Dextran (bisher 70 kDa, R_h = 6,5 nm angenommen)?
**Stelle:** `FRAP/frap_analysis_v2.py`, `DEXTRAN_RH_NM` (Stokes-Einstein-Referenz für H); Docstrings von `frap_analysis_v2.py` und `frap_figures.py` nennen noch 70 kDa.
**Wirkung heute:** D_SE = 32,97 µm²/s (R_h = 6,5 nm, 20 °C); H gegen Stokes-Einstein (D_propagation, 20 mg/mL, n = 9) = 0,822. Gemessenes Wasser: 30,13 µm²/s, H gegen Wasser = 0,900. D_eff/D0 in `frap_depth_figures.py` nutzt nur das gemessene Wasser und ist nicht betroffen.
- A: R_h für 150 kDa vom Nutzer angeben (Quelle: Hersteller oder Literatur) → D_SE sinkt (D_SE ∝ 1/R_h), H gegen Stokes-Einstein steigt; Neulauf von `frap_analysis_v2.py`.
- B: Stokes-Einstein-Referenz entfällt, nur H gegen gemessenes Wasser (n = 2) → keine R_h-Annahme, aber Referenz aus nur zwei Messungen.
**Entscheidung:** A, R_h = 9,0 ± 0,6 nm, doi:10.1002/jps.23387 (Nutzer, 2026-10-07). Neulauf: D_SE = 23,81 µm²/s, H gegen Stokes-Einstein (D_propagation, 20 mg/mL, n = 9) = 1,138 (95 %-KI 1,108–1,174).

## E017 · ENTSCHIEDEN · 2026-10-08
**Frage:** FRAP v2: Soll das Kriterium „Messdauer ≥ 5 t_half“ für die Recovery-Auswertung bleiben?
**Stelle:** `FRAP/frap_analysis_v2.py`, `QC_MIN_DURATION_THALF = 5.0` (QC-Gate `qc_recovery`, kein Vermerk einer Nutzerentscheidung); ebenso ohne Vermerk: `QC_MIN_RECOVERY_R2 = 0.95`, `QC_MIN_FRAMES_TO_THALF = 10`.
**Wirkung heute:** 3 von 13 Messungen fallen nur an diesem Kriterium durch: water_1 (32 s < 35 s = 5 × 7,1 s), water_3 (52 s < 64 s = 5 × 12,9 s), interface_1 (15 s < 103 s = 5 × 20,6 s); R² ≥ 0,989 und ≥ 19 Frames bis t_half erfüllen alle drei. Folge: t_half, τ, M_f und D_Soumpasis/Axelrod/Kang werden ohne diese drei gemittelt; damit besteht keine Wassermessung, H gegen Wasser ist für diese Methoden nicht berechenbar, Hydrogel shallow hat n = 1. Nicht betroffen: D_propagation, D_profile, D_eff/D0 und die Recovery-Kurven in `frap_depth_figures.py` (Gruppenmittel nutzen `qc_global`). water_2 fällt separat an gesättigten Pixeln (97,2 % > 1 %) durch und ist vom Nutzer ausgeschlossen (2026-10-06).
- A: Kriterium bleibt (5 t_half) → wie heute.
- B: Schwelle senken, Wert vom Nutzer → water_1 (4,5 t_half) und water_3 (4,0 t_half) bestehen ab Schwelle ≤ 4; interface_1 (0,7 t_half) besteht erst ohne Kriterium.
- C: Kriterium streichen, R² und Frames bis t_half bleiben → alle 13 bestehen `qc_recovery`; Wasser n = 2, Hydrogel shallow n = 2 für die Recovery-Parameter; M_f aus Kurven, die das Plateau nicht erreicht haben, geht ein.
**Entscheidung:** Nein, das ist trenge Kriterium ist nicht nötig, der Exponentiale Fit passt bei allen Daten. Lediglich die Übersättigten Pixel bei Water2 sorgen für Ausschluss. (Nutzer, 2026-10-08)

## E018 · ENTSCHIEDEN · 2026-10-08
**Frage:** FRAP v2: Welche Methode liefert D für D_eff/D0 und die Abbildungen – Propagation (heute) oder Axelrod?
**Stelle:** `FRAP/frap_analysis_v2.py` (`D_axelrod_um2_s = 0.88 r_e² / (4 t_half)`), `FRAP/frap_depth_figures.py` (`D_COLUMN`, Nutzer 2026-10-06: `D_propagation_um2_s`).
**Wirkung heute:** D = D_propagation (formfrei, ganzes Profil). Der Code-Faktor 0,88 bei Axelrod ist der Faktor für eine gleichmäßige Kreisscheibe; für einen Gaußstrahl hängt γ_D bei Axelrod (1976) von der Bleichtiefe K ab. D_axelrod ist an `qc_recovery` gebunden: heute besteht keine Wassermessung (E017). D_axelrod Hydrogel deep 17,2–35,2 µm²/s, steigt mit r_e (28–66 µm, r_e größer bei längerer Bleichphase 1,9–11,2 s); D_propagation derselben Messungen 25,6–29,6 µm²/s. Wasser: Axelrod 41,5 / 26,5, Propagation 30,6 / 29,7.
- A: Propagation bleibt → keine Änderung.
- B: Axelrod wie im Code (0,88) → D0 erst berechenbar, wenn E017 Wasser zulässt; ohne QC-Filter D_eff/D0 Hydrogel deep 0,784, shallow 0,764 (heute 0,900 / 0,908).
- C: Axelrod mit γ_D(K) für den Gaußstrahl aus der Originalarbeit → neue Implementierung, zuerst an synthetischen Daten zu prüfen; D steigt gegenüber B um γ_D / 0,88.
**Entscheidung:** Propagation bleibt in den bestehenden Abbildungen; das PyFRAP-D (E019) liefert alternative Abbildungen und eine Vergleichsgrafik beider Methoden in einem weiteren Skript (Nutzer, 2026-10-08).

## E019 · Geschlossen· 2026-10-08
**Frage:** FRAP v2: Soll der nominelle Bleichradius r_n aus der Leica-ROI übernommen werden statt r_n = 0 (Punktbleiche)?
**Stelle:** `FRAP/measurements.csv` (`r_nominal_um = 0.0` für alle), `FRAP/frap_analysis_v2.py` (`D_kang_um2_s`, `D_soumpasis_nominal_um2_s`), `FRAP/frap_figures.py` (Beschriftung „point bleach, r_n = 0“).
**Wirkung heute:** Laut Properties-XML (LAS X 3.5.7, FRAP-Wizard, Zoom In, kein Fly-Mode) wurde in allen 17 Messungen eine Kreis-ROI gebleicht, Größe 19,2 × 19,4 µm (Water 1–3, B1_4–6, noFITC_1) bzw. 32,6 × 32,6 µm (übrige); vermutlich Durchmesser, r_n ≈ 9,6 bzw. 16,3 µm. Mit r_n = 0 ist D_Kang je Messung 6–13 % niedriger als mit diesem r_n (z. B. gel_B1_1 12,6 → 14,2 µm²/s). D_propagation, D_profile, Axelrod, Soumpasis und D_eff/D0 nutzen r_n nicht. Register 0.2: „nominal bleach ROI is never a geometric input“ (Nutzer).
- A: r_n = 0 bleibt → wie heute; Beschriftung „point bleach“ widerspricht den Metadaten.
- B: r_n aus der XML-ROI (halbe Größe) → D_Kang +6 bis +13 %; Kang wie in der Originalarbeit (r_n und r_e); Ausnahme vom Registereintrag nur für Kang.
- C: r_n = 0 bleibt, nur die Beschriftung wird auf „Kreis-ROI, r_n nicht verwendet“ korrigiert.
**Entscheidung:** Ja es soll der bleichradius übernommen werden - der Ort kann aber durch drift verändert sein und muss in jedem Frame angepasst werden. nutze pyFRAP um das D zu bestimmen.

## E020 · OFFEN · 2026-10-08
**Frage:** FRAP/PyFRAP: Wie wird die Bleich-ROI (r_n aus der XML, E019) pro Frame an den Drift angepasst, und wo liegt sie in Frames, in denen die Bleichstelle nicht mehr erkennbar ist?
**Stelle:** neues Skript (PyFRAP-Auswertung), Extraktion der ROI-Kurven.
**Wirkung heute:** Das Bleichzentrum ist über den Schwerpunkt des Defizits nur verfolgbar, solange die zentrale Absenkung ≥ 5 % ist: in 5 von 13 Messungen bis zum letzten Frame, sonst bis 19,8 s (water_1) bis 134,6 s (gel_B1_3) bei Messdauern bis 554,6 s; max. Versatz 1,2–11,1 µm. Die XML-Position der ROI stimmt bei den 32,6-µm-ROIs auf ≤ 3 px mit dem gemessenen Zentrum überein, bei den 19,2-µm-ROIs (Water 1–3, B1_4–6) liegt sie 51 µm (≈ 184 px) in x daneben; Position daher aus den Daten, nur der Radius aus der XML.
- Nachführung: A: ROI-Maske wandert pro Frame mit dem Zentrum, Bilder bleiben unverändert (PyFRAP rechnet nur Simulation und Fit, die Kurven kommen aus dem neuen Skript). B: Bilder pro Frame um ganze Pixel verschoben, danach feste PyFRAP-ROIs (PyFRAP-eigene Extraktion; Bilddaten werden verschoben, Randpixel fallen weg).
- Späte Frames: A: letzte erkannte Position halten. B: Drift linear über die erkannten Frames anpassen und fortschreiben. C: Messung nur bis zum letzten erkannten Frame auswerten (bis zu 420 s Daten fallen weg).
**Entscheidung:** der Peak der Bleiche wird per Gauß gefittet udn die das ROI darübergelegt. Falls das nicht geht, bleibt das ROI an der Stelle.


## E021 · OFFEN · 2026-10-08
**Frage:** FRAP/PyFRAP: Mit welcher 3D-Geometrie und welcher Bleichform in z wird simuliert?
**Stelle:** neues Skript, PyFRAP `embryo.geometry`, `sliceDepthMu`, `sliceWidthMu`, `simulation.ICmode`.
**Wirkung heute:** PyFRAP simuliert in 3D und braucht Probengeometrie (z. B. Zylinder: Radius, Höhe), Tiefe der Bildebene und Dicke des optischen Schnitts; dafür gibt es keine Defaults (Abschnitt 0.1). Vorhanden sind nur die Tiefen 306 µm (planar verankert) bzw. 10 µm (shallow, Nutzer). Objektiv 63×/1,30 Glycerin, Pinhole 1 AU. ICmode 3 (PyFRAP-Standard) überträgt das erste Post-Bleach-Bild in z unverändert (Bleichsäule durch die ganze Höhe); ICmode 2 dämpft die Bleiche in z mit einer Sigmoidfunktion (Parameter nötig).
- Werte vom Nutzer: Probenradius, Gelhöhe, Schnittdicke.
- Bleichform: A: ICmode 3, Säule durch die ganze Höhe. B: ICmode 2, Sigmoid in z (Parameter vom Nutzer). C: ICmode 4, ideale Scheibe mit r_n (ohne Bildinformation).
**Entscheidung:** Zylinder ist Plausibel. es hanelt sich um ein Voxel, die z Höhe ist in den Parametern des mikroskops zu finden. die Bleichvolumen ist viel kleiner als das Medium dick ist. 


## E022 · ENTSCHIEDEN · 2026-10-08
**Frage:** FRAP/PyFRAP: Welche z-Höhe hat das Bleichvolumen, und wie groß wird der simulierte Zylinder gewählt?
**Stelle:** neues Skript (PyFRAP-Simulation), Anfangsbedingung in z und `embryo.geometry` (Zylinder, E021).
**Wirkung heute:** Die einzige z-Angabe in den Metadaten ist `OptimalStepSize` = 0,333 µm (`Sections="1"`, kein z-Stapel; das ist die vom System vorgeschlagene Schrittweite, keine gemessene Höhe). Eine 0,333 µm dicke gebleichte Schicht würde sich bei D ≈ 30 µm²/s in z innerhalb von Millisekunden (≈ h²/D ≈ 4 ms) auffüllen; gemessen ist im ersten Frame 0,37 s nach dem Bleichen noch eine Absenkung K0 = 0,55–0,90 vorhanden. PyFRAP-ICmode 2 (Bleiche in z begrenzt) ist laut Quelltext „Not working in current version“. Medium ≫ Bleichvolumen (Nutzer): der Zylinder ist dann ein Rechengebiet, keine Probengröße; shallow liegt 10 µm über dem Glas.
- z-Höhe: A: 0,333 µm aus der XML → widerspricht der gemessenen Absenkung, Fit liefert kein sinnvolles D. B: Höhe h als zweiter freier Parameter neben D (eigene Schleife um den PyFRAP-Fit). C: Wert vom Nutzer (z. B. aus einem xz-Scan). D: Säule durch das ganze Rechengebiet (ICmode 3).
- Zylindergröße: A: so groß, dass D sich bei doppeltem Radius/Höhe um < 1 % ändert (Konvergenztest, Wert dokumentiert). B: Werte vom Nutzer. Shallow: Glas als Boden 10 µm unter der Bildebene.
**Entscheidung:** z-Höhe vorläufig 1 µm (Nutzer, 2026-10-08: „erstmal“). der Bleichradius aus der xml datei. Rechengebiet: Würfel statt Zylinder, Kantenlänge aus der Fokusebene (Bildfeld x/y, 144,72 µm), z nicht größer; Bleiche und Messung bleiben in der Fokusebene, z nur als Pseudo-Dimension (Nutzer, 2026-10-08). PyFRAP hat keinen eingebauten Würfel, er wird als eigene .geo-Datei angelegt. Geändert: PyFRAP rechnet mit seinem Zylinder statt Würfel; Normierung nach PyFRAP-Standard ((Bild + 1)/(Pre-Bleach-Bild + 1)) (Nutzer, 2026-10-08). Höhe des Zylinders: E024.

## E023 · ENTSCHIEDEN · 2026-10-08
**Frage:** FRAP/PyFRAP: Wann gilt der Gauß-Fit der Bleichstelle als „nicht gegangen“ (E020), wenn der Fleck breiter als das Bildfeld wird?
**Stelle:** `FRAP/frap_pyfrap_inputs.py`, `fit_bleach` (heute nur: keine Konvergenz, K ≤ 0, w ≤ 0, Zentrum außerhalb des Bildes).
**Wirkung heute:** Bis K ≈ 0,4 stimmen Gauß-Zentrum und Schwerpunkt aus v2 auf ≤ 2 µm überein. Wird w ≳ 60–100 µm (Bildfeld 145 µm), wandert das Fit-Zentrum: max. Versatz water_1 65 µm (v2-Schwerpunkt 11 µm), gel_B1_2 43 µm, gel_B1_3 37 µm; gescheitert sind nur 19 von 1090 Frames (water_1 17, gel_B1_2 2). Wirkung auf die Recovery-Kurve (Skala 0–1), Gauß-ROI gegen feste ROI am ersten Zentrum: water_1 max. 0,166, gel_B1_3 0,031, gel_B1_2 0,024, interface_2 0,010.
- A: wie heute → ROI folgt dem Fit auch bei sehr breitem Fleck.
- B: Fit gilt als gescheitert, wenn w größer als ein Anteil des Bildfelds ist (Wert vom Nutzer) → ROI bleibt danach stehen.
- C: Fit gilt als gescheitert, wenn das Zentrum zwischen zwei Frames mehr als eine Strecke springt (Wert vom Nutzer).
- D: Fit gilt als gescheitert, wenn K unter einer Schwelle liegt (Wert vom Nutzer).
**Entscheidung:** D mit 5 %: Fit gilt als gescheitert, wenn die Lokalisierung nicht möglich ist oder der Fit zu flach ist (K < 0,05); dann bleibt die ROI stehen (Nutzer, 2026-10-08).

## E024 · ENTSCHIEDEN · 2026-10-08
**Frage:** FRAP/PyFRAP: Wie hoch ist der Zylinder im Verhältnis zur 1-µm-Bleichschicht – hoch (145 µm, Bleiche nur in der Fokusebene) oder so hoch wie die Bleichschicht (quasi-2D)?
**Stelle:** neues PyFRAP-Skript (Stufe 2), `embryo.geometry` (Zylinder), Gitter und Anfangsbedingung.
**Wirkung heute:** Test mit PyFRAP/gmsh: Zylinder r = 72,4 µm, h = 144,7 µm, Gitter in der Schicht ±2,8 µm um die Fokusebene auf 0,57 µm verfeinert → 2,52 Mio. Zellen, Gitter 205 s, ein Zeitschritt ≈ 53 s; PyFRAP-Standard 3000 Schritte ≈ 44 h je Messung (13 Messungen). Ohne Verfeinerung (25 836 Zellen, ≈ 7 µm) wird die 1-µm-Schicht nicht aufgelöst. Physikalisch füllt sich eine 1 µm dünne Schicht in z in ≈ h²/D ≈ 33 ms auf; die Daten zeigen 0,37 s nach dem Bleichen noch 55–90 % Absenkung. PyFRAP-Standard (ICmode 3) ist eine Bleichsäule über die ganze Höhe. `addBoxField` von PyFRAP ist unter Python 3 fehlerhaft (bytes/str); Umgehung über eine eigene .geo-Kopie.
- A: Zylinderhöhe = Bleichhöhe 1 µm (Bleiche füllt die ganze Höhe, keine Diffusion in z, „z nur Pseudo“) → kleines Gitter, Minuten je Messung; entspricht PyFRAP-Standard ICmode 3 in einem flachen Zylinder.
- B: Zylinder 144,7 µm hoch, Bleiche 1 µm, Gitter in der Schicht grob (≈ 1 µm Zellen) → Stunden je Messung, Schicht nur mit ≈ 1 Zelle aufgelöst; D wird von der Diffusion in z bestimmt.
- C: Zylinder 144,7 µm hoch, Bleichsäule über die ganze Höhe (PyFRAP-Standard ICmode 3) → Minuten je Messung; Bleiche nicht auf die Fokusebene beschränkt (gleiche Physik wie A).
**Entscheidung:** A: Höhe in z 1 µm, nichts weiter; alle übrigen Parameter darf Claude festlegen (dokumentiert im Skript). Zusätzlich eine Version rein in 2D, weil in 2D gemessen und gebleicht wurde (Nutzer, 2026-10-08).

## E025 · ENTSCHIEDEN · 2026-10-08
**Frage:** Rheologie-Kernabbildungen: Welche Läufe gehen ein, welche Unsicherheit wird angegeben, welche Probe zeigt die Viskoelastizität, was zeigt die Stabilitätsgrafik und welche Altersgruppe geht in die Porengröße ein?
**Stelle:** `Rheology/rheo_hyd20_key_figures.py` (`SELECTION`, `CI_LEVEL`, `REPRESENTATIVE_AGE_D`, `RANDOM_SEED`, `POOL_MAX_AGE_D`, `FUNCTIONALITIES`, `STRAND_DIAMETER_RANGE_NM`).
**Wirkung heute:** qc_filtered schließt die Codes axial_force, presheared, cracks und group_outlier sowie FAIL-Läufe aus (13 von 27 Läufen); unter 10 d gehen damit 4 Proben ein (4 d: 3, 9 d: 1), G' = 773 ± 145 Pa (95 %-KI). all_analysable hätte 6 Proben (676 ± 175 Pa SD).
- Auswahl: A: qc_filtered. B: qc_filtered ohne group_outlier. C: all_analysable.
- Unsicherheit: A: SD. B: SEM. C: 95 %-KI (t-Verteilung) über Proben. D: SD und SEM.
- Grafik 1: A: mittlerer Frequenzsweep < 10 d + Amplitudensweeps 9 d. B: nur mittlerer Frequenzsweep. C: eine repräsentative Probe.
- Grafik 2: A: G' und G'' bei 0,5 Hz. B: zusätzlich tan δ. C: G' normiert auf 4 d.
**Entscheidung:** Auswahl A (qc_filtered wie bisher, damit E008 A); Unsicherheit C (95 %-KI, t-Verteilung, N = Proben, kein Intervall bei N = 1); Grafik 1 C mit einer zufällig gezogenen 4-d-Probe (Seed 20261008 → Guss 27.10., A3, Hyd20 (8); bei 4 d gibt es keinen Amplitudensweep); Grafik 2: nur G' bei 0,5 Hz gegen das Alter; Porengröße: alle Proben unter 10 d gepoolt (4 d und 9 d), Phantomnetzwerk mit f = 4, 6, 8, Fibrillendurchmesser 9–13 nm (Nutzer, 2026-10-08).

## E026 · ENTSCHIEDEN · 2026-10-08
**Frage:** Rheologie: Sollen trägheitsdominierte Messpunkte (Trägheitsanteil > 1) und Punkte mit Drehmoment < 5·10⁻⁸ N m weiter aus den Fits genommen werden?
**Stelle:** `Rheology/Rheo_Hyd20_Compute.py`, `POINT_CRITERIA` (`point_artifacts` in `rheo_analysis.py`); nicht im Register (Abschnitt 0.2).
**Wirkung heute:** In 14 Frequenzsweeps fallen 1 bis 4 Punkte am Hochfrequenzende aus Plateaufenster und G'-Fit (alle „inertia-dominated“, kein Drehmoment-Ausschluss); die Punkte bleiben in Tabellen und werden grau gezeigt. Zählung je Datei in `rheo_hyd20_key_figures_exclusions.csv`.
- A: behalten → Kriterium ins Register, im Methodenteil nennen.
- B: abschaffen → Fits nutzen alle Punkte; trägheitsverfälschte Hochfrequenzpunkte gehen in Fenster und G'_rep ein (Neulauf von `Rheo_Hyd20_Compute.py`).
- C: andere Schwelle für den Trägheitsanteil → Wert vom Nutzer.
**Entscheidung:** A, behalten (Trägheitsanteil > 1, Drehmoment < 5·10⁻⁸ N m), ins Register, im Methodenteil nennen (Nutzer, 2026-10-08).

## E027 · ENTSCHIEDEN · 2026-10-08
**Frage:** Rheologie-Kernabbildungen, Überarbeitung: Welcher Amplitudensweep ergänzt die Viskoelastizität, wie geht der Fibrillendurchmesser in den Fehler der Porengröße ein, was zeigt die Porengrafik, und was zeigt die Mesh-Size-Grafik?
**Stelle:** `Rheology/rheo_hyd20_key_figures.py` (`AMPLITUDE_SWEEP_AGE_D`, `STRAND_DIAMETER_COVERAGE`, `plot_cavity_vs_f`, `plot_mesh_size`).
**Wirkung heute:** Bei 4 d gibt es keinen Amplitudensweep; qc_filtered-Amplitudensweeps bei 9 d: Hyd20, Hyd20 (1), Hyd20 (2). Beispiel f = 4, d_f = 11 nm: Kavität 17,6 nm, 95 %-KI ± 1,9 nm, d_f-Spanne ± 2,0 nm.
- Amplitudensweep: A: 13-d-Probe mit beiden Sweeps (Läufe 20–24). B: Frequenzsweep 4 d A3 bleibt, Amplitudensweep zufällig aus 9 d. C: alle qc-Messungen.
- Fehler d_f: A: linear addiert (± 3,9 nm). B: quadratisch mit 2 nm (± 2,7 nm). C: d_f gleichverteilt 9–13 nm, 1,96 · 4/√12 = 2,26 nm, quadratisch mit dem KI (± 2,9 nm).
- Porengrafik (x = f): A: Kavität und Throat. B: nur Kavität.
- Mesh Size (affin, ξ = (k_B T/G')^(1/3), f-unabhängig): A: ξ gegen Alter. B: ξ < 10 d gepoolt mit 95 %-KI und Einzelproben. C: A mit Band.
**Entscheidung:** Amplitudensweep B (Seed 20261008 → Hyd20, 9 d, unbeschriftete Probe); Fehler d_f C; Porengrafik B (Throat nur in den Tabellen); Mesh Size B. Stabilitätsgrafik nur qc-Läufe, ohne Mittelwerte, Legende „Hydrogel 20 mg mL⁻¹“, Achsen „G' (Pa) at 0.5 Hz“ und „t (days) after Gel Cast“; Gitterzellen (f = 4, 6, 8) als eigene Abbildung (Nutzer, 2026-10-08).

## E028 · ENTSCHIEDEN · 2026-10-08
**Frage:** Rheologie-Kernabbildungen, zweite Überarbeitung: Wie werden Viskoelastizität, Stabilität, Gitterdarstellung, Kavität gegen f und Mesh Size dargestellt?
**Stelle:** `Rheology/rheo_hyd20_key_figures.py` (`plot_viscoelasticity`, `plot_stability`, `plot_cells`, `plot_cavity_vs_f`, `AS_YLIM_PA`, `STABILITY_MARKERSIZE`).
**Wirkung heute:** Mit y = 10–600 Pa liegen 2 von 42 Punkten des Amplitudensweeps Hyd20 außerhalb (G' = 5,8 und 2,0 Pa bei den größten Dehnungen) und sind nicht sichtbar.
- Legende und Stern im Amplitudensweep: A: beide weglassen. B: beide behalten. C: Legende weg, Stern bleibt.
- Mesh Size für die Caption: A: Proben < 10 d (N = 4, 17,5 ± 1,1 nm). B: alle qc-Proben (N = 8, ≈ 16,4 ± 1,1 nm).
**Entscheidung:** Viskoelastizität nur Amplitudensweep (Hyd20), ohne Titel, schwarz, Legende und Stern A (weglassen), y 10–600 Pa; Stabilität ohne Legende, Symbole kleiner; Gitter ohne Fibrillendicke (Kavität und Throat mit Achsendurchmessern), auf eine Kavität zentriert und periodisch in alle Richtungen fortgesetzt; Kavität gegen f in Schwarz; Mesh Size keine Abbildung, Wert A für die Caption (Nutzer, 2026-10-08). Nachtrag: Gitter als beleuchtetes 3D-Rendering (pyvista/VTK) mit glatter, halbtransparenter Kavitätskugel, Nachbarzellen nur angedeutet, Stränge vor der Kavität blass, näher am Objekt (Nutzer, 2026-10-08). Nachtrag (Nutzer, 2026-10-14): Viskoelastizität mit Legende G′ (gefüllt) / G″ (offen) und ohne die senkrechten Linien bei γ5% und γ10%; ersetzt für diese Abbildung „Legende weglassen“ (die Dehnungswerte bleiben in Tabelle 1).

## E029 · ENTSCHIEDEN · 2026-10-22
**Frage:** FRAP-Bildserie (Hydrogel oben, Wasser unten, t = 0/10/20 s): Welche Messung wird je Reihe gezeigt?
**Stelle:** `FRAP/frap_recovery_image_panels.py`, `SELECTION_RULE` (nutzt `_representative` aus `FRAP/frap_figures.py`).
**Wirkung heute:** Regel aus Fig. 1 von `frap_figures.py` übernommen: QC-bestandene (Profil) Messung mit der größten Bleichtiefe K0 → Hydrogel gel_B1_1 (K0 = 0,81, Kandidaten: 9 Bulk-Gel-Messungen A1/B1, 20 mg/mL), Wasser water_3 (K0 = 0,72; water_2 fällt an gesättigten Pixeln durch, water_1 hat K0 = 0,53). Für diese Regel ist keine Nutzerentscheidung vermerkt.
- A: Regel „größtes K0“ beibehalten → Abbildung bleibt wie erzeugt (gel_B1_1, water_3).
- B: Messungen vom Nutzer benannt (z. B. eine typische statt der tiefsten Bleiche) → nur die gezeigten Bilder ändern sich, keine Zahlenwerte.
**Entscheidung:** A (Nutzer, 2026-10-22)

## E030 · ENTSCHIEDEN · 2026-10-22
**Frage:** Ist die Pixelgröße 0,130 µm/px für 800 px breite Filme (Binning ×1) richtig, oder müsste sie nach dem Binning 0,075 µm/px sein?
**Stelle:** `core.io` (Auswertung der `.rec`-Datei, Zeile 147: `x == 800 → mpp = 0.130`, Kommentar „mpp measured via Thorlab Grid 10µm“); gleiche Tabelle `MPP_BY_IMAGE_WIDTH_PX` in `Validation_Claude/snr_signal_profile.py`.
**Wirkung heute:** Dieselbe Kamera (pco.pixelfly USB, Seriennummer 1401) hat am 30.01.2026 `500 nm_2.tif` mit 400 × 300 px, Binning ×2 (→ 0,15 µm/px) und drei Minuten später `500 nm_3.tif` mit 800 × 600 px, Binning ×1 aufgenommen. Bei 0,30 µm/px für Binning ×4 und 0,15 µm/px für Binning ×2 ergäbe Binning ×1 0,075 µm/px. Betroffen: immobilisierte 560-nm-Filme `500 nm_3` und `500 nm_4` (42 Tracks in Abb20; FWHM-Median 1092 nm gegen 659 nm bei 0,15 µm/px und 736 nm bei 0,30 µm/px; mit 0,075 µm/px wären es 1092 × 0,075/0,130 ≈ 630 nm) und die Gruppe 560 nm / 0,13 µm/px der Lokalisationsfehler-Tabelle (σ_loc 50 ± 16 nm). Die Umrechnung 0,15 gegen 0,30 µm/px (240/560/1370 nm gegen 35–100 nm) ist dagegen mit dem Binning konsistent. Abb20 zeigt die Werte unverändert.
- A: 0,130 µm/px bleibt (eigene Kalibrierung mit dem 10-µm-Gitter für diese Einstellung) → keine Änderung; Kalibrieraufnahme im Methodenteil nennen.
- B: 0,075 µm/px aus dem Binning → `core.io` ändern, `Particle_Size_SNR_Compute.py` und `Loc_Error_Analyse_immob_particle.py` neu laufen lassen; 560-nm-FWHM und σ_loc der zwei Filme sinken um den Faktor 0,58.
- C: die beiden 800-px-Filme aus Abb20 und der σ_loc-Tabelle herausnehmen, bis die Pixelgröße geklärt ist → 560 nm nur aus 222 Tracks bei 0,15 und 0,30 µm/px.
**Entscheidung:** B -> da der Ausschnitt der gleiche bleibt (Nutzer, 2026-10-22)

## E031 · ENTSCHIEDEN · 2026-10-22
**Frage:** Wie werden mehrfach analysierte Filme in den MSD-Caches behandelt (Buffer `msd_d0_results.pkl` und Hydrogel `msd_20mg_files.pkl`)?
**Stelle:** `MSD_FromTrackmate_D0.py` und `MSD_FromTrackmate_20mg.py` (Dateisuche); Folgeskripte Abb05, Abb08–Abb12, Abb14, Abb15, Abb17–Abb19.
**Wirkung heute:** Buffer 35 nm (nominal 20 nm) zählt 9 Einträge, aber nur 4 verschiedene Filme: `20 nm water_4` viermal (je eine Kopie in `D_0 Wassermessung\20 nm\Tracks` und `2026.01.16\Tracks_20` mit identischem Ergebnis D = 13,06 µm²/s, dazu `filtered_20 nm water_4` D = 13,01 und `Resultof20 nm water_4` D = 11,79), `20 nm_water_02` und `20 nm_water_03` je zweimal (beide Ordner, identische Ergebnisse), dazu `20 nm_water` (0,15 µm/px, 40 fps, D = 1,6). Hydrogel 35 nm zählt 21 Einträge für 19 Filme: `20nm_1d_B4_04` und `20nm_1d_B4_04_processed` (D = 3,05 / 2,95), `20nm_1d_A4_01` und `Resultof20nm_1d_A4_01` (D = 2,94 / 6,27). Alle Einträge gehen heute einzeln in Abbildungen und gewichtete Mittel ein; nichts wurde entfernt.
- A: so lassen → jede Analyse zählt als eigene Messung; Doppelungen im Methodenteil nennen.
- B: je Film nur eine Analyse behalten (Regel vom Nutzer, z. B. „Originalordner, unbearbeitete Tracks“) → Buffer 35 nm 4 statt 9 Werte, Hydrogel 35 nm 19 statt 21; Neulauf der Compute-Skripte und aller Folgeabbildungen.
- C: identische Kopien (gleiches Ergebnis, zwei Ordner) entfernen, bearbeitete Varianten (`filtered_`, `Resultof`, `_processed`) behalten → Buffer 35 nm 6 statt 9.
**Entscheidung:** nur eine analyse behalten - aber die andern werden in dem Summary behandelt unter andere Methode. (Nutzer, 2026-10-22)

## E032 · ENTSCHIEDEN · 2026-10-22
**Frage:** Welche DLS-Referenzwerte gelten in den SPT-Abbildungen (D₀ und Größenfehler)?
**Stelle:** `core.io.get_dls_reference_maps()` (fest eingetragene Werte) gegen `core.io.get_dls_diffusion_means()` / `get_dls_sizes()` (DLS-Cache, laut Docstring maßgeblich); benutzt in Abb09–Abb12.
**Wirkung heute:** Die x-Position kommt aus dem Cache (z. B. 557,9 und 1366,9 nm), D₀, D₀-Fehler und Größenfehler aus den fest eingetragenen Werten, die zu älteren Durchmessern gehören (621,5 und 1266,5 nm). Für 35–240 nm stimmen beide überein; für 560 nm D₀ = 0,700 (fest) gegen 0,774 µm²/s (Cache), für 1370 nm 0,341 gegen 0,314 µm²/s. Dadurch weicht die DLS-Referenz in Abb12 bei 560/1370 nm um −8,9 / +8,8 % von Stokes–Einstein ab, bei 35–240 nm um 0,0 %. Größenfehler fest 21,8 / 47,0 nm gegen DLS-Tabelle (Abb35) ± 51,1 / ± 38,3 nm.
- A: so lassen → Abbildungen unverändert; Mischung im Methodenteil begründen.
- B: alle Referenzwerte aus dem DLS-Cache (`get_dls_diffusion_means()`, Größenfehler aus derselben Auswertung) → DLS-Punkte 560/1370 nm verschieben sich, Abb09 unverändert (nur 35/50 nm).
**Entscheidung:** B und die Maps verbessern (Nutzer, 2026-10-22)

## E033 · ENTSCHIEDEN · 2026-10-22
**Frage:** Welche Messungen sind für 560 nm (nominal 500 nm) die „drei Referenzmessungen“ in Abb35/Abb36?
**Stelle:** `Litesizer/litesizer_visualization_chapter.py` (über `litesizer_visualization.group_accepted_measurements`), Quelle `Size_repitition_All Sizes.xlsx`.
**Wirkung heute:** Für 500 nm enthält die Referenzdatei 12 Messungen, davon 6 mit PDI ≤ 42 % (zwei Serien „500 nm 1–3“: d_H 500,8 / 558,3 / 533,8 nm und 516,5 / 620,7 / 617,6 nm). Alle 6 werden gezeigt und gemittelt (d_H 557,9 ± 51,1 nm); alle anderen Größen haben genau 3. Nichts wurde entfernt.
- A: alle 6 behalten → wie heute; im Text „3 bzw. 6 Wiederholungen“ nennen.
- B: nur eine Serie (vom Nutzer benannt) → 3 Messungen; d_H, D₀ und PDI für 560 nm ändern sich und damit die x-Position von 560 nm in allen SPT-Abbildungen, wenn der DLS-Cache neu erzeugt wird.
**Entscheidung:**alle 6 nehmen (Nutzer, 2026-10-22)

## E034 · ENTSCHIEDEN · 2026-10-22
**Frage:** Welche Feret-Definition zeigen Abb31 (50 nm) und Abb32 (240 nm) gemeinsam?
**Stelle:** `SEM_Particles/sem_feret_size_histogram.py` (`DIAMETER_VARIANTS`); ausgewählt sind derzeit `sem_feret_diameter_50nm.pdf` (Feret_max) und `sem_mean_feret_diameter_200nm.pdf` (mittlerer Feret = (Feret_max + Feret_min)/2).
**Wirkung heute:** Beide Varianten werden für jede Größe erzeugt. 50 nm: Feret_max 39,4 ± 8,4 nm, mittlerer Feret 36,6 ± 8,4 nm; 240 nm: Feret_max 244,5 ± 17,8 nm, mittlerer Feret 228,2 ± 20,1 nm (n = 1000 / 1010, ohne Pt-Abzug). Abb33 nutzt Feret_max.
- A: Feret_max für beide → Abb32 als `sem_feret_diameter_200nm.pdf`; einheitlich mit Abb33.
- B: mittlerer Feret für beide → Abb31 als `sem_mean_feret_diameter_50nm.pdf`; Abb33 nutzt dann eine andere Definition, außer sie wird ebenfalls umgestellt.
- C: zusätzlich Pt-Abzug (2 × 1,3 nm wie in Abb33) auch in Abb31/32 → Werte um 2,6 nm kleiner.
**Entscheidung:** Ferret max für beide (Nutzer, 2026-10-22)

## E035 · ENTSCHIEDEN · 2026-10-22
**Frage:** Wie wird die nicht normierte FRAP-Intensität (FITC-Messungen gegen noFITC-Kontrollen) in der Vergleichsgrafik gezeigt?
**Stelle:** neues Skript `FRAP/frap_raw_intensity_comparison.py` (Consumer von `cache/frap_v2_phase1.pkl`, Pre-Bleach-Frames aus den TIFs wie in `frap_depth_figures.py`).
**Wirkung heute:** Alle Abbildungen zeigen nur doppelt normierte Kurven. Rohwerte (Mittel in der Bleich-ROI, 8 bit, Hintergrund 0) vor dem Bleichen: Buffer 61,6 / 92,1 (water_2 gesättigt, 254,6, vom Nutzer ausgeschlossen); Hydrogel 28,8–84,6; noFITC 15,6 / 83,1 / 68,7 Counts. Detektor (PMT 1, Gain 800, Offset 0) und Pinhole (1 AU) sind überall gleich, die Aufnahme-Laserleistung nicht: 1,0 % bei water_1–3, nofitc_1, gel_B1_1–6, gel_A1_2/3; 0,5 % bei nofitc_2/3, interface_1/2, gel_A1_1. nofitc_3 hat Zoom 1 statt 1,275 (0,361 statt 0,283 µm/px).
- Messgröße: A: Mittel in der Bleich-ROI (Radius r_e der Messung; Kontrollen an der mittleren Bleichstelle, r = 46,4 µm, wie in v2). B: Mittel über das ganze Bildfeld. C: beide als zwei Panels.
- Laserleistung: A: alle Messungen wie aufgenommen, Laserleistung im Symbol/Linienstil markiert, keine Umrechnung. B: nur Messungen mit 1,0 % in der Grafik (übrige in der Tabelle). C: zwei Panels, getrennt nach 0,5 % und 1,0 %.
- Darstellung: A: Zeitverlauf jeder Messung (Pre-Bleach und Recovery) in Counts. B: wie A plus Gruppenmittel je Probe und Laserleistung. C: nur Pre-Bleach-Niveau je Messung als Punkt (kein Zeitverlauf).
**Entscheidung:** Einzelne Grafiken Pro Bleaching - in einem Subfigure Raster, nur für den Anhang und zur Übersicht. Im Dialog: Messgröße A (Bleich-ROI), Laserleistung A (alle Messungen, markiert, keine Umrechnung), Darstellung A (Zeitverlauf in Counts) (Nutzer, 2026-10-22).

## E036 · ENTSCHIEDEN · 2026-10-26
**Frage:** Welcher Unsicherheitsbereich wird um den Amsden-Fit von D_eff/D0 (20 mg/mL, r_f = 6 nm) schattiert?
**Stelle:** `MSD_Trackmate/MSD_Deff_over_D0_vs_Size_20mg_Amsden.py`, `BAND_K` und `fit_amsden` (gewichteter Fit, absolute Fehler).
**Wirkung heute:** Fit über 2 Punkte (35 nm: 0,178 ± 0,032; 50 nm: 0,095 ± 0,015, 1 SE) mit einem freien Parameter: r̄ = 11,18 ± 0,51 nm, χ² = 4,94 bei 1 Freiheitsgrad. Einzeln invertiert ergeben die Punkte r̄ = 9,78 nm (35 nm) und 12,05 nm (50 nm). Gezeigt ist ±1σ mit absoluten Fehlern (r̄ 10,68–11,69 nm); der 35-nm-Punkt liegt mit seinem Fehlerbalken außerhalb des Bandes.
- A: ±1σ, absolute Fehler (heute) → schmales Band, gibt nur die Fehlerbalken wieder, nicht die Streuung zwischen den Punkten.
- B: ±1σ, σ mit √χ²_red skaliert → r̄ 10,06–12,31 nm; Band deckt beide Einzelpunkt-Werte ab.
- C: alle r̄, deren Kurve jeden Fehlerbalken einzeln trifft (Einhüllende der Einzelpunkte ± 1 SE) → r̄ 8,95–12,62 nm; breitestes der Bänder A–C.
- D: 95 %-KI (t-Verteilung, 1 Freiheitsgrad, skaliertes σ) → r̄ −3,1 bis 25,5 nm; 12,7-mal so breit wie B (t-Faktor bei 1 Freiheitsgrad).
**Entscheidung:** A, B und D, alle drei als geschachtelte Bänder in einer Abbildung (Nutzer, 2026-10-26). Am selben Tag geändert: nur A (Nutzer, 2026-10-26).

## E037 · ENTSCHIEDEN · 2026-10-26
**Frage:** Der Amsden-Fit mit freiem Fibrillenradius (3 nm ≤ r_f ≤ 15 nm, Nutzer) endet an der Obergrenze r_f = 15 nm – wie wird damit umgegangen?
**Stelle:** `MSD_Trackmate/MSD_Deff_over_D0_vs_Size_20mg_Amsden.py`, `R_F_MODE = "free"`, `R_F_BOUNDS_NM`, `fit_amsden`.
**Wirkung heute:** χ² fällt über den ganzen Bereich stetig mit r_f (r_f = 3 nm: 7,96; 6 nm: 4,94; 12 nm: 1,81; 15 nm: 1,03). Der Fit liegt deshalb auf der Grenze: r_f = 15 nm, r̄ = 7,79 ± 0,65 nm (σ bei festgehaltenem r_f, da ein Parameter auf der Grenze keinen symmetrischen Fehler hat). Ohne Grenze läuft die Kurve exakt durch beide Punkte (χ² = 0, 0 Freiheitsgrade) bei r_f = 29,2 nm, r̄ = 2,2 nm. Bei 2 Punkten und 2 freien Parametern ist der Fehler nicht verlässlich: An synthetischen Daten (wahr r̄ = 10, r_f = 8 nm) deckt ±1σ r̄ nur in 54,7 % der Fälle ab (erwartet 68,3 %), und 849 von 2000 Fits enden auf einer Grenze. Gezeigt in `deff_over_d0_vs_size_20mg_amsden_rf_free.pdf`; die Version mit r_f = 6 nm (`..._amsden.pdf`) bleibt bestehen.
- A: so lassen → r_f = 15 nm auf der Grenze, r̄ = 7,79 ± 0,65 nm; im Text angeben, dass r_f an der Grenze liegt.
- B: Grenze erweitern (Wert vom Nutzer) → ab r_f ≥ 29,2 nm exakter Durchgang, r̄ = 2,2 nm, kein Fehler schätzbar.
- C: r_f fest (Wert vom Nutzer, z. B. 6 nm wie bisher oder aus der Rheologie 4,5–6,5 nm) → r̄ nur aus einem Parameter, z. B. 11,18 ± 0,51 nm bei 6 nm.
- D: kein Fit von r_f; nur das χ²-Profil über r_f (Tabelle in `..._rf_free_fit_results.txt`) berichten.
**Entscheidung:** C mit r_f = 1,5 nm fest; gesucht ist der Porendurchmesser 2r̄ (Nutzer, 2026-10-26). Ergebnis: r̄ = 12,78 ± 0,44 nm, 2r̄ = 25,56 ± 0,88 nm, χ² = 10,09 (1 Freiheitsgrad).

## E038 · ENTSCHIEDEN · 2026-10-26
**Frage:** In welcher Form wird das Amsden-Modell für D_eff/D0 gefittet?
**Stelle:** `MSD_Trackmate/MSD_Deff_over_D0_vs_Size_20mg_Amsden.py`, `amsden_ratio`; abweichende Form in `MSD_Trackmate/MSD_Normalize_Hydrogel_vs_Water.py`, Origin `Amsden.FDF`/`Amsden_v2.FDF` und Hydrogel-Network-Simulation `amsden_fit.py`.
**Wirkung heute:** Das Skript nutzte exp[−π/4 ((r_s + r_f)/(r̄ + r_f))²]; das ist dieselbe Funktion wie Amsden 2022 Gl. 3 mit ξ = 2r̄. Die übrigen Stellen nutzen exp[−π((r_s + r_f)/(ξ + r_f))²] (r_f statt 2r_f im Nenner); dort gilt ξ_Amsden = ξ_alt − r_f (gemessen, r_f = 4 nm: 28,9 → 24,9 nm).
- A: Amsden 2022 Gl. 3, exp[−π((r_s + r_f)/(ξ + 2r_f))²], ξ = Mesh Size als Fitparameter.
- B: bisherige Form der übrigen Stellen.
**Entscheidung:** A, die neuere Form (Amsden 2022, Gl. 3; nach Amsden 1999 und Hadjiev & Amsden 2015) (Nutzer, 2026-10-26). Ergebnis bei r_f = 1,5 nm: ξ = 25,56 ± 0,88 nm, χ² = 10,09 (1 Freiheitsgrad). Die übrigen Stellen sind nicht geändert.

## E039 · ENTSCHIEDEN · 2026-10-27
**Frage:** Soll D_eff aus dem MSD um den Lokalisationsfehler (und die Bewegungsunschärfe) korrigiert werden?
**Stelle:** `core.analysis.perform_msd_analysis` und `fit_powerlaw_with_errors`: D = A/4 aus dem log-log-Fit MSD = A·τ^n über die ersten 6 Lags, ohne Offset-Term; σ_loc aus dem linearen Fit MSD = 4Dτ + 4σ² wird nur berichtet, nicht abgezogen. Belichtungszeit wird aus der .rec gelesen, aber nicht verwendet.
**Wirkung heute:** Hydrogel 20 mg/mL, Achsenabschnitt des linearen Fits relativ zum MSD beim ersten Lag (1/60 s): 35 nm Median 0,16 (−0,22 bis 1,14, 21 Dateien), 50 nm Median 0,65 (0,32 bis 0,77, 12 Dateien). Gewichtete D_eff/D0 (DLS): Potenzgesetz A/4 (heute) 35 nm 0,178, 50 nm 0,095; Steigung des linearen Fits/4: 35 nm 0,206, 50 nm 0,182. Exponent heute 0,89 (35 nm) und 0,62 (50 nm). Der Achsenabschnitt des linearen Fits enthält neben 4σ² auch Krümmung durch Subdiffusion, er ist kein reiner Lokalisationsfehler. Die Amsden-Abbildung (E036–E038) beruht auf den heutigen Werten. Verwandt: E005 (Berichten von σ_loc), E009 (Drift).
- A: so lassen → D = A/4 aus dem Potenzgesetz ohne Offset; im Methodenteil nennen.
- B: Fit MSD = 4D·τ^n + 4σ² mit σ frei (3 Parameter auf 6 Lags) → D, n und σ gemeinsam; Neulauf der Compute-Skripte.
- C: σ aus den immobilisierten Partikeln (`Validation_Claude/Loc_Error_*`) je Bedingung fest abziehen, dann Potenzgesetz → unabhängige σ-Quelle; braucht σ für 35/50 nm bei den verwendeten Pixelgrößen.
- D: linearer Fit MSD = 4Dτ + 4σ² (n = 1 angenommen) → D/D0 35 nm 0,206, 50 nm 0,182.
- Zusätzlich je Option: Bewegungsunschärfe über die Belichtungszeit berücksichtigen (ja/nein).
**Entscheidung:** C und dazu die Bewegungsunschärfe mit 4D(τ- 1/3 t_E) + 4σ²  , t_E entspricht Belichtungszeit. Im Dialog präzisiert (Nutzer, 2026-10-27): Modell MSD = 4D(τ − t_E/3)^n + 4σ², σ = Werte der immobilisierten Partikel für die entsprechende Pixelgröße (Mittel von `sigma_xy_static_nm` je Größe und mpp: 35 nm/0,3 µm/px 118,8 nm, 50 nm/0,3 µm/px 76,1 nm). Ergebnis Hydrogel (Gewicht E002 A): 35 nm D = 1,79 ± 0,30 µm² s⁻ⁿ, n = 0,87 ± 0,04; 50 nm D = 0,66 ± 0,10, n = 0,57 ± 0,03; D_eff/D0 0,146 / 0,078; Amsden (r_f = 1,5 nm) ξ = 23,73 ± 0,75 nm, χ² = 15,9.

## E040 · ENTSCHIEDEN · 2026-10-27
**Frage:** Sollen E002, E003, E006, E009, E031 und E039 auch in der Wasserauswertung (`MSD_FromTrackmate_D0.py`, D0 der SPT) gelten?
**Stelle:** `MSD_Trackmate/MSD_FromTrackmate_D0.py` (Cache `msd_d0_results.pkl`), Folgeskripte Abb12, Abb23, `MSD_Normalize_Hydrogel_vs_Water.py`, Abb03/07, Abb17–19.
**Wirkung heute:** Der Hydrogel-Cache ist seit 2026-10-27 mit Lokalisationsfehler und Bewegungsunschärfe gefittet, der Wasser-Cache noch mit dem alten Potenzgesetz (A/4, Gewicht `num_tracks`, keine Ausschlussliste). Vergleiche Hydrogel gegen SPT-Wasser mischen damit zwei Methoden; D_eff/D0 mit DLS-D0 (Abb09, Amsden) ist nicht betroffen. Hindernisse für E006 A und E039 C im Wasser: `1000_nm_10_Tracks_v2.xml` (Ordner `2026.01.19\Tracks_1000`) hat keine .rec; für 1370 nm bei 0,3 µm/px (`1000_nm_01` bis `_09`) gibt es kein σ der immobilisierten Partikel (nur 0,15 µm/px: 10,4 nm). Doppelanalysen (E031) im Wasser: `20 nm water_4` viermal, `20 nm_water_02`/`_03` je zweimal.
- A: alle genannten Entscheidungen auch im Wasser → Neulauf von `MSD_FromTrackmate_D0.py`; braucht die .rec zu `1000_nm_10_Tracks_v2` und ein σ (Wert und Quelle) für 1370 nm bei 0,3 µm/px.
- B: Wasser bleibt wie heute → Methodenmischung im Methodenteil nennen.
- C: im Wasser alles außer E039 (ohne σ- und Unschärfekorrektur) → kein σ für 1370 nm nötig; Methodenmischung beim Fitmodell bleibt.
**Entscheidung:** A: alle SPT-Auswertungen werden im Wasser gleich getätigt, selbe Funktion, selbe Parameter, selbe Unsicherheiten (Nutzer, 2026-10-27). Umgesetzt: gemeinsames Modul `MSD_Trackmate/msd_pipeline.py`, das beide Compute-Skripte aufrufen. Wasser: 39 Filme ausgewertet, 6 weitere Analysen (3 identische Kopien, 3 Re-Analysen) nur in `msd_d0_per_file.csv`.

## E041 · ENTSCHIEDEN · 2026-10-27
**Frage:** Welches σ der immobilisierten Partikel gilt für 1370 nm (nominal 1000 nm) bei 0,3 µm/px, wofür es keine immobilisierte Aufnahme gibt?
**Stelle:** `MSD_Trackmate/msd_pipeline.py`, `SIGMA_MPP_SUBSTITUTES`.
**Wirkung heute:** betrifft 9 Wasserfilme (`1000_nm_01` bis `_09`); immobilisiert vorhanden nur 0,15 µm/px (10,4 nm, 49 Tracks, `1000nm.xml`, `1k.xml`). 4σ²/MSD(τ₁) dort 0,004 bis 0,014.
- A: Neue Messung abwarten → Wasserlauf bricht ab.
- B: σ von 0,15 µm/px übernehmen → 10,4 nm, Ausnahme protokolliert.
- C: Wert vom Nutzer.
**Entscheidung:** B (Nutzer, 2026-10-27). Jede betroffene Datei steht mit `sigma_loc_substitute` in `msd_d0_exclusions.csv`.

## E042 · ENTSCHIEDEN · 2026-10-27
**Frage:** Welche Pixelgröße der immobilisierten Partikel bestimmt das feste σ im MSD-Fit, und wie wird der Lokalisierungsfehler der bewegten Proben ausgegeben?
**Stelle:** `MSD_Trackmate/msd_pipeline.py`, `SIGMA_MPP_BY_SIZE`, `_sigma_for()`; `core/analysis.py`, `locerror_from_msd_slope()`.
**Wirkung heute:** σ wurde je Datei nach (Größe, mpp der Datei) gewählt; abweichend nur `20 nm water` bei 0,15 µm/px (σ 35,6 nm) und `1000_nm_01` bis `_09` bei 0,3 µm/px (E041). Ein dynamischer Lokalisierungsfehler wurde nicht ausgegeben.
- A: σ nach mpp der Datei (bisher) → wie oben.
- B: σ nach Größe: 0,3 µm/px für nominal 20, 50, 100 nm, 0,15 µm/px für 200, 500, 1000 nm → `20 nm water` (0,15 µm/px) bekommt 118,9 nm statt 35,6 nm; 1000 nm unverändert 10,4 nm (ersetzt E041).
**Entscheidung:** B (Nutzer, 2026-10-27; 100 nm → 0,3 µm/px). Jede Datei, deren eigene Pixelgröße abweicht, steht mit `sigma_loc_mpp` in der Ausschlussliste. Zusätzlich pro Datei die Lokalisierungsfehler-Varianz der bewegten Probe (Nutzer, 2026-10-27): Gerade durch die ersten 6 eMSD-Punkte, Steigung = 4D, zum Achsenabschnitt wird (4/3)·D·t_E addiert, der Rest ist 4σ² (Savin & Doyle 2005); ausgegeben neben σ² der immobilisierten Partikel (`sigma2_loc_dynamic_nm2`, `sigma2_loc_static_nm2`, `msd_offset_var_nm2` = Achsenabschnitt/4) in `<prefix>_per_file.csv` und gewichtet pro Größe in `<prefix>_weighted_average.csv`. Nur Ausgabe, geht nicht in den Fit ein.

## E043 · ENTSCHIEDEN · 2026-10-27
**Frage:** Mit welchen Größen wird der dynamische Lokalisierungsfehler nach Gl. 4.6 der Dissertation, δ_dyn = δ_sta·√(1 + D·t_E/s₀²), pro Datei berechnet?
**Stelle:** `core/physics.py`, `calculate_dynamic_localization_error()`; `MSD_Trackmate/msd_pipeline.py`, `PSF_SIGMA_S0_NM`, `analyse_file()`.
**Wirkung heute:** Bisher nur die Varianz aus dem Achsenabschnitt der MSD-Geraden (E042); Gl. 4.6 nicht ausgewertet. PSF-Fits der immobilisierten Partikel liegen nicht für jede Größe bei der Pixelgröße aus E042 vor (500 nm nur bei 0,3 µm/px, 20 nm bei 0,3 µm/px nur 12 Tracks).
- s₀: A: PSF-Fit bei der Pixelgröße aus E042 → 500 nm mit dem 0,3-Wert; B: PSF-Fit über alle Pixelgrößen; C: Theorie 0,21·λ/NA bzw. fester Wert → ein s₀ für alle Größen.
- D: D des Pipeline-Fits (Einheit µm² s⁻ⁿ) oder D_lin = Steigung/4 der Geraden durch die ersten 6 eMSD-Punkte (µm²/s).
**Entscheidung:** s₀ = 250 nm für alle Größen und Medien; D = D_lin; δ_sta = festes σ der immobilisierten Partikel nach E042 (Nutzer, 2026-10-27). Ausgabe `delta_dyn_eq46_nm`, `delta_dyn2_eq46_nm2` in `<prefix>_per_file.csv`, gewichtet pro Größe in `<prefix>_weighted_average.csv`. Nur Ausgabe, geht nicht in den Fit ein.

## E044 · ENTSCHIEDEN · 2026-10-27
**Frage:** Was soll Abb03 (Korrektur-Abweichung vorher/nachher) zeigen, seit der Pipeline-Fit selbst σ und Bewegungsunschärfe enthält?
**Stelle:** `MSD_Trackmate/Validation_Claude/Figures_Refined/Correlations_Refined.py` (`A_correction_deviation_refined.pdf`), `Validation_Claude/Correlations.py`, `correct_and_refit()`.
**Wirkung heute:** „vorher“ ist jetzt der korrigierte Pipeline-Fit (E039 C), „nachher“ die alte Methode (4σ² vom eMSD abziehen, Potenzgesetz neu fitten, σ = `_representative_sigma_loc_nm`). Die Abbildung vergleicht damit zwei Korrekturen statt unkorrigiert gegen korrigiert. Letzter Lauf 2026-10-27 13:12.
- A: vorher = unkorrigierter Fit aus dem Cache (`D_uncorrected`, `exponent_uncorrected`), nachher = Pipeline-Fit → Abb03 zeigt die Wirkung der jetzigen Korrektur; Skript wird angepasst.
- B: Abb03 entfällt → Verweis im Text auf die Tabellen (`D_uncorrected` neben `D_um2_per_s`).
- C: unverändert lassen → Beschriftung „before/after“ ist nicht mehr zutreffend.
**Entscheidung:** A (Nutzer, 2026-10-27). Umgesetzt in `Correlations_Refined.py`; Werte pro Datei in `A_correction_deviation_refined_per_file.csv`; Hydrogel-Boxen kreuzschraffiert.

## E045 · ENTSCHIEDEN · 2026-10-27
**Frage:** Sollen die Fits pro Track (iMSD) in Abb14–15 und Abb17–19 ebenfalls mit Lokalisierungsfehler und Bewegungsunschärfe gefittet werden?
**Stelle:** `Validation_Claude/Fittingpoints_robustness.py`, `per_track_fits_by_points()` (Abb14–15); `Validation_Claude/TaskB1_TrackLengthBias_Compute.py` → `cache/track_length_bias_*.pkl` (Abb17–19, n-Histogramm).
**Wirkung heute:** beide fitten jede iMSD-Spalte mit dem Potenzgesetz `fit_powerlaw_with_errors` ohne σ und t_E. Abb14–15 wurde am 2026-10-27 aus den neuen Caches gerechnet (Tracks wie Pipeline), Abb17–19 liest `track_length_bias_*.pkl` vom 2026-09-23 (Dateistand und Filter von damals, ohne E031/E003).
- A: beide auf `fit_powerlaw_blur_locerror` mit dem σ aus E042 umstellen, TaskB1 neu rechnen → Track-n und Track-D mit derselben Korrektur wie die eMSD-Fits.
- B: unkorrigiert lassen, nur TaskB1 mit den heutigen Dateien neu rechnen → Methodenunterschied Track gegen Ensemble im Text nennen.
- C: unverändert lassen.
**Entscheidung:** A (Nutzer, 2026-10-27). Umgesetzt; ausgelassene Tracks (NaN/≤ 0 im Fenster, Fit nicht konvergiert) in `SPT/track_length_bias/track_length_bias_exclusions.csv` und `SPT/fittingpoints_exclusions.csv`.

## E046 · ENTSCHIEDEN · 2026-10-27
**Frage:** Welche Form hat der Term für Lokalisierungsfehler und Bewegungsunschärfe im MSD-Fit bei n ≠ 1?
**Stelle:** `core/analysis.py`, `msd_blur_locerror_general()`, `fit_powerlaw_blur_locerror_general()`, `perform_msd_analysis_blur_locerror(model=…)`; `MSD_Trackmate/msd_pipeline.py`, `MSD_MODEL`; Fits pro Track in `TaskB1_TrackLengthBias_Compute.py` und `Fittingpoints_robustness.py`.
**Wirkung heute:** Fit mit MSD = 4D(τ − t_E/3)ⁿ + 4σ² (E039 C); diese Form ist nur für n = 1 exakt. Im Hydrogel ist n ≈ 0,57 bis 0,87.
- A: 4D(τ − t_E/3)ⁿ + 4σ² beibehalten.
- B: allgemeine Form (Savin & Doyle 2005, Backlund et al. 2015), 2D: MSD(τ) = 4K/[(n+1)(n+2)·t_E²]·[(τ+t_E)^(n+2) + |τ−t_E|^(n+2) − 2τ^(n+2) − 2t_E^(n+2)] + 4σ².
**Entscheidung:** B (Nutzer, 2026-10-27). K wird als `D_um2_per_s` gespeichert (Einheit µm² s⁻ⁿ), Modell in `fit_results_MSD["model_id"] = "general"`. Für n = 1 identisch mit A (geprüft, relative Abweichung < 1e-15). Gilt für Wasser, Hydrogel und die Fits pro Track.

## E047 · OFFEN · 2026-10-27
**Frage:** Wie sollen die Fits pro Track mit dem allgemeinen Modell (E046) umgehen, das nur für n > 0 definiert ist (bei τ = t_E ist der Klammerterm t_E^(n+2)·(2^(n+2) − 4), für n ≤ 0 also ≤ 0)?
**Stelle:** `core/analysis.py`, `fit_powerlaw_blur_locerror_general()` (ohne Grenzen für n); `Validation_Claude/TaskB1_TrackLengthBias_Compute.py` (Abb17–19), `Validation_Claude/Fittingpoints_robustness.py` (Abb14–15).
**Wirkung heute:** Fits pro Track, die nicht konvergieren oder am Startwert scheitern, werden ausgelassen und protokolliert: TaskB1 Wasser 465 von 6915 Tracks (35 nm 40, 50 nm 116, 100 nm 108, 240 nm 8, 560 nm 38, 1370 nm 155), Hydrogel 195 von 1722 (35 nm 77, 50 nm 118 von 539); Abb14–15 2344 (Wasser) und 1046 (Hydrogel) Kombinationen aus Track und Punktzahl. Es fallen bevorzugt Tracks mit kleinem n weg. Die Ensemble-Fits pro Datei sind nicht betroffen (alle konvergiert).
- A: Fit pro Track mit Grenzen 0 < n < 2 (beschränkter Optimierer) → jeder Track bekommt einen Wert, Tracks am Rand werden markiert und gezählt.
- B: wie heute, gescheiterte Tracks auslassen → Histogramme und Mediane ohne diese Tracks.
- C: pro Track das bisherige Modell 4D(τ − t_E/3)ⁿ + 4σ² (für jedes n definiert) → Methodenunterschied Track gegen Ensemble.
**Entscheidung:** offen

## E048 · OFFEN · 2026-10-27
**Frage:** FRAP v2: Wie wird der Hintergrund F_BG begründet bzw. gewählt, nachdem nur noch eine noFITC-Kontrolle (nofitc_1) übrig ist?
**Stelle:** `FRAP/frap_analysis_v2.py`, `BACKGROUND_COUNTS = 0.0` und Modul-Docstring („The noFITC controls cannot serve as a quantitative background (their intensity scatters by a factor of 5 …)“).
**Wirkung heute:** F_BG = 0 in allen Messungen. Die Begründung „Streuung um Faktor 5“ stammt aus den drei Kontrollen 15,6 / 83,1 / 68,7 Counts; nofitc_2 und nofitc_3 sind seit 2026-10-27 ausgeschlossen (FITC vorhanden, Nutzer), es bleibt nofitc_1 mit 15,6 Counts in der Bleich-ROI vor dem Bleichen (1,0 % Laser). FITC-Messungen vor dem Bleichen: 28,8–92,1 Counts; 3 von 13 gezeigten (interface_1, interface_2, gel_A1_1) mit 0,5 % Laser, dafür gibt es keine Kontrolle mehr. Laut Docstring kürzt sich ein konstanter Untergrund aus τ, t_half, M_f und w(t) und senkt nur das scheinbare K0.
- A: F_BG = 0 beibehalten, Docstring nur mit PMT-Offset 0 begründen → keine Zahl ändert sich.
- B: F_BG = Niveau von nofitc_1 (15,6 Counts) für die 1,0-%-Messungen → K0 und F_norm ändern sich; für die 0,5-%-Messungen ist ein eigener Wert nötig (vom Nutzer).
- C: F_BG = 0, Abzug von nofitc_1 nur als Diagnose in den Tabellen → Hauptergebnisse unverändert, zusätzliche Spalten.
**Entscheidung:** offen

## E049 · OFFEN · 2026-10-27
**Frage:** FRAP: Welche Unsicherheit wird für D_eff/D0 angegeben und gespeichert (Abbildung `frap_deff_d0_vs_depth.pdf` und Tabelle `frap_deff_d0_values.csv`)?
**Stelle:** `FRAP/frap_depth_figures.py`, `plot_ratio()` und `write_ratio_table()`.
**Wirkung heute:** Gezeigt und gespeichert werden der Mittelwert ± SD (ddof = 1) von D_eff/D0 über die QC-bestandenen Messungen: Hydrogel 0,902 ± 0,042 (n = 11, flach und tief zusammen), Buffer 1,000 ± 0,020 (n = 2). D0 = 30,13 µm²/s (water_1, water_3) mit einer SD von 0,62 µm²/s geht als Konstante ein, seine Unsicherheit wird nicht fortgepflanzt. Für D_propagation gibt es je Messung keine Fit-Unsicherheit, deshalb haben die Einzelwerte in `frap_depth_values.csv` keine. `frap_analysis_v2.py` speichert daneben H = geometrisches Mittel D_gel / D_Wasser (nur tiefes Gel, N = 9) = 0,900 mit 95 %-Bootstrap-KI [0,869; 0,933] in `ratio_gel_water.json`.
- A: wie heute, SD über die Messungen, D0 als Konstante → Hydrogel ± 0,042.
- B: SD über die Messungen, die relative SD von D0 quadratisch addiert (Gauß) → Hydrogel ± 0,046.
- C: Standardfehler des Mittelwerts statt SD → Hydrogel ± 0,013 (ohne D0) bzw. 95 %-t-KI ± 0,028; Buffer (n = 2) 95 %-t-KI ± 0,184.
**Entscheidung:** offen
