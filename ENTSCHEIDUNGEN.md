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

## E001 · OFFEN · 2026-10-06
**Frage:** Randfilter behalten? Detektionen in den äußeren 3 % des Bildes werden gelöscht.
**Stelle:** `core.io.remove_edge_artifacts`, läuft automatisch in `single_file_data`; im Code seit 2026-07-07.
**Wirkung heute:** entfernt in Wasser 0,5 bis 9,6 % der Detektionen je Partikelgröße, in einzelnen Dateien über 50 %; Tracks werden an der Lücke geteilt.
- A: behalten (3 %) → im Methodenteil begründen, Zahl der entfernten Detektionen je Bedingung berichten.
- B: abschalten → mehr Detektionen und längere Tracks; mögliche Fehlverknüpfungen am Rand bleiben in den Daten.
- C: Sensitivität rechnen (0, 1, 3 %) und danach festlegen → ein zusätzlicher Durchlauf, zeigt, ob D davon abhängt.
**Entscheidung:** offen

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

## E007 · OFFEN · 2026-10-06
**Frage:** DLS-Messungen mit Polydispersität über 42 % ausschließen?
**Stelle:** `Litesizer/*`, `MAX_PDI = 42.0`.
**Wirkung heute:** Messungen über der Schwelle fehlen in Größen und D₀ aus der DLS; die Zahl wird nur in der Konsole genannt.
- A: behalten → Schwelle begründen, Zahl der ausgeschlossenen Messungen je Größe berichten.
- B: alle Messungen zeigen und die auffälligen markieren → kein Ausschluss, breitere Streuung.
- C: andere Schwelle → mit Quelle oder Herstellerangabe festlegen.
**Entscheidung:** offen

## E008 · OFFEN · 2026-10-06
**Frage:** Rheologie: Läufe mit dem Kennzeichen `group_outlier` (Faktor 2 gegen den Median gleich alter Läufe) aus der gefilterten Statistik lassen?
**Stelle:** `Rheology/Rheo_Hyd20_Compute.py`, `STATS_EXCLUDE_QC_CODES`, `QC_GROUP_OUTLIER_FACTOR`.
**Wirkung heute:** Diese Läufe bleiben in Tabellen und Abbildungen, fehlen aber in der Statistik `qc_filtered`.
- A: behalten → Kriterium im Methodenteil nennen.
- B: nur markieren, nicht ausschließen → Statistik über alle gültigen Läufe.
**Entscheidung:** offen

## E009 · OFFEN · 2026-10-06
**Frage:** Driftkorrektur im MSD-Code: entfernen, behalten oder richtig einführen?
**Stelle:** `core.analysis.perform_msd_analysis`, Bedingung „Partikel je Frame > 50“.
**Wirkung heute:** keine. Die Bedingung wurde bei keiner der 77 Dateien ausgelöst (höchster Wert 1,1), es wird also nie Drift abgezogen. Der Code steht aber da und würde bei dichteren Filmen still eingreifen.
- A: entfernen → der Code tut, was im Methodenteil steht: keine Driftkorrektur.
- B: behalten → als Regel dokumentieren, obwohl sie nie greift.
- C: Drift prüfen und dann entscheiden → einmal die mittlere Verschiebung je Film ausgeben lassen.
**Entscheidung:** offen
