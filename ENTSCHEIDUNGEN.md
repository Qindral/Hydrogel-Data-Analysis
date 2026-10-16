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
**Entscheidung:** Rand ist immer von unsicherheiten tracking linking betroffen und muss ausgeschlossen werden.
muss nicht weiter begründet werden. 

## E002 · OFFEN · 2026-10-06
**Frage:** Womit werden die Dateien beim Mittelwert je Partikelgröße gewichtet?
**Stelle:** `core.analysis.weighted_average_per_size`, Gewicht = `num_tracks`.
**Wirkung heute:** `num_tracks` wird vor dem 10-Frame-Filter gezählt. Bei 50 nm im Hydrogel sind 66 % dieser Tracks zu kurz und gehen nicht in den Fit ein, bestimmen aber das Gewicht.
- A: Trackzahl nach dem 10-Frame-Filter → Gewicht entspricht den Tracks, die wirklich im Fit stecken.
- B: ungewichtet, jede Datei zählt gleich → passt zur Regel „das Movie ist die Einheit“; Dateien mit wenigen Tracks wiegen dann so viel wie große.
- C: so lassen → D und Fehlerbalken bleiben wie bisher, die Gewichtung ist dann zu begründen.
**Entscheidung:** offen

## E003 · OFFEN · 2026-10-06
**Frage:** Mindest-Bildrate von 40 Hz für nominal 20, 50 und 100 nm behalten?
**Stelle:** `MSD_FromTrackmate_D0.py` und `MSD_FromTrackmate_20mg.py`, `MIN_FPS_SMALL_SIZES`.
**Wirkung heute:** unbekannt. Übersprungene Dateien werden nur in der Konsole genannt und nirgends gespeichert. Zuerst zählen lassen, welche Dateien betroffen sind.
- A: behalten → übersprungene Dateien in die Ausschlussliste schreiben und im Methodenteil nennen.
- B: abschaffen → alle Dateien gehen ein; langsam aufgenommene Filme kleiner Partikel haben wenige, große Schritte.
- C: andere Schwelle → mit Begründung über Schrittweite je Frame festlegen.
**Entscheidung:** offen

## E004 · OFFEN · 2026-10-06
**Frage:** Wie viele Lag-Punkte gehen in den MSD-Fit: 6 (Standard im Code) oder 4 (in den Validierungsläufen benutzt)?
**Stelle:** `core.analysis.DEFAULT_MSD_FIT_POINTS = 6`.
**Wirkung heute:** D und α im Kapitel stammen aus 6 Punkten, die Validierung zum Tracklängen-Bias aus 4.
- A: 6 Punkte überall → Validierung mit 6 wiederholen.
- B: 4 Punkte überall → Hauptauswertung neu rechnen.
- C: Sensitivität 2 bis 6 Punkte einmal berichten und einen Wert festlegen → beantwortet die Frage auch für die Prüfer.
**Entscheidung:** offen

## E005 · OFFEN · 2026-10-06
**Frage:** Wie wird der Lokalisationsfehler berichtet, wenn der Achsenabschnitt der MSD negativ ist?
**Stelle:** `core.analysis.perform_msd_analysis`, `sigma_loc_nm`; Mittelung in `weighted_mean_and_err`.
**Wirkung heute:** Ein negativer Achsenabschnitt wird zu „kein Wert“ und fällt aus dem Mittel. In Wasser betrifft das 22 von 44 Dateien, im Hydrogel 3 von 33. Der gemittelte Wert stammt also nur aus Dateien mit positivem Achsenabschnitt.
- A: σ aus den immobilisierten Partikeln als Hauptwert, Achsenabschnitt nur als Diagnose zeigen → kein Auswahleffekt.
- B: über die Achsenabschnitte aller Dateien mitteln (auch negative), erst danach die Wurzel ziehen → Mittel ohne Auswahl, kann nahe null oder negativ ausfallen.
- C: so lassen → im Text angeben, dass nur positive Achsenabschnitte eingehen und wie viele Dateien das sind.
**Entscheidung:** offen

## E006 · OFFEN · 2026-10-06
**Frage:** Was passiert mit Dateien ohne Kalibrierung (.rec fehlt, Bildbreite unbekannt) oder mit fehlgeschlagenem Fit?
**Stelle:** `core.io.single_file_data` (gibt `None` zurück), `core.analysis.analyze_multiple_files` (fängt Fehler ab).
**Wirkung heute:** Die Datei wird übersprungen, es gibt nur eine Konsolenmeldung. In den gespeicherten Ergebnissen ist nicht zu sehen, ob und welche Dateien fehlen.
- A: Abbruch mit Fehlermeldung, die Datei und fehlendes Feld nennt → keine Datei kann unbemerkt fehlen; der Lauf stoppt, bis das Problem behoben ist.
- B: überspringen, aber in die Ausschlussliste schreiben → der Lauf geht durch, das Fehlen ist dokumentiert.
**Entscheidung:** offen

## E007 ·Geschlossen · 2026-10-06
**Frage:** DLS-Messungen mit Polydispersität über 42 % ausschließen? 
**Stelle:** `Litesizer/*`, `MAX_PDI = 42.0`.
**Wirkung heute:** Messungen über der Schwelle fehlen in Größen und D₀ aus der DLS; die Zahl wird nur in der Konsole genannt.
- A: behalten → Schwelle begründen, Zahl der ausgeschlossenen Messungen je Größe berichten.
- B: alle Messungen zeigen und die auffälligen markieren → kein Ausschluss, breitere Streuung.
- C: andere Schwelle → mit Quelle oder Herstellerangabe festlegen.
**Entscheidung:** Ja ab eine Polydispersität ist das Ergebnis zu unsicher unter 0.3 sehr genau. 6-10-2026

## E008 · ENTSCHIEDEN · 2026-10-08
**Frage:** Rheologie: Läufe mit dem Kennzeichen `group_outlier` (Faktor 2 gegen den Median gleich alter Läufe) aus der gefilterten Statistik lassen?
**Stelle:** `Rheology/Rheo_Hyd20_Compute.py`, `STATS_EXCLUDE_QC_CODES`, `QC_GROUP_OUTLIER_FACTOR`.
**Wirkung heute:** Diese Läufe bleiben in Tabellen und Abbildungen, fehlen aber in der Statistik `qc_filtered`. Betroffen: Hyd20 (26) (13 d, G' = 2865 Pa gegen Median 1104 Pa).
- A: behalten → Kriterium im Methodenteil nennen.
- B: nur markieren, nicht ausschließen → Statistik über alle gültigen Läufe.
**Entscheidung:** A, Statistik mit `qc_filtered` wie bisher, zusammen mit den Codes axial_force, presheared und cracks (Nutzer, 2026-10-08, siehe E025).

## E009 · OFFEN · 2026-10-06
**Frage:** Driftkorrektur im MSD-Code: entfernen, behalten oder richtig einführen? Driftkorrektur nur bei Wasser.
**Stelle:** `core.analysis.perform_msd_analysis`, Bedingung „Partikel je Frame > 50“.
**Wirkung heute:** keine. Die Bedingung wurde bei keiner der 77 Dateien ausgelöst (höchster Wert 1,1), es wird also nie Drift abgezogen. Der Code steht aber da und würde bei dichteren Filmen still eingreifen.
- A: entfernen → der Code tut, was im Methodenteil steht: keine Driftkorrektur.
- B: behalten → als Regel dokumentieren, obwohl sie nie greift.
- C: Drift prüfen und dann entscheiden → einmal die mittlere Verschiebung je Film ausgeben lassen.
**Entscheidung:** offen

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
**Entscheidung:** Viskoelastizität nur Amplitudensweep (Hyd20), ohne Titel, schwarz, Legende und Stern A (weglassen), y 10–600 Pa; Stabilität ohne Legende, Symbole kleiner; Gitter ohne Fibrillendicke (Kavität und Throat mit Achsendurchmessern), auf eine Kavität zentriert und periodisch in alle Richtungen fortgesetzt; Kavität gegen f in Schwarz; Mesh Size keine Abbildung, Wert A für die Caption (Nutzer, 2026-10-08). Nachtrag: Gitter als beleuchtetes 3D-Rendering (pyvista/VTK) mit glatter, halbtransparenter Kavitätskugel, Nachbarzellen nur angedeutet, Stränge vor der Kavität blass, näher am Objekt (Nutzer, 2026-10-08).

## E029 · OFFEN · 2026-10-14
**Frage:** FRAP-Bildserie (Hydrogel oben, Wasser unten, t = 0/10/20 s): Welche Messung wird je Reihe gezeigt?
**Stelle:** `FRAP/frap_recovery_image_panels.py`, `SELECTION_RULE` (nutzt `_representative` aus `FRAP/frap_figures.py`).
**Wirkung heute:** Regel aus Fig. 1 von `frap_figures.py` übernommen: QC-bestandene (Profil) Messung mit der größten Bleichtiefe K0 → Hydrogel gel_B1_1 (K0 = 0,81, Kandidaten: 9 Bulk-Gel-Messungen A1/B1, 20 mg/mL), Wasser water_3 (K0 = 0,72; water_2 fällt an gesättigten Pixeln durch, water_1 hat K0 = 0,53). Für diese Regel ist keine Nutzerentscheidung vermerkt.
- A: Regel „größtes K0“ beibehalten → Abbildung bleibt wie erzeugt (gel_B1_1, water_3).
- B: Messungen vom Nutzer benannt (z. B. eine typische statt der tiefsten Bleiche) → nur die gezeigten Bilder ändern sich, keine Zahlenwerte.
**Entscheidung:** offen
