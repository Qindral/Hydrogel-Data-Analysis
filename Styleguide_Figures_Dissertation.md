# STYLEGUIDE: Scientific Figures in Python (Microscopy + Plots) – Dissertation

Version 2 – angepasst an das LaTeX-Layout der Dissertation (A4, 2,5 cm Rand, 11 pt Fließtext, Captions 10 pt).

---

## 0) Grundprinzipien

- Ziel: ruhige, präzise, journal- und dissertationstaugliche Figuren; die Daten stehen visuell im Vordergrund.
- **Goldene Regel: Jede Figur wird in exakt der Größe erzeugt, in der sie im PDF erscheint, und ohne Skalierung eingebunden.** Nur dann sind Schrift, Ticks und Linien in allen Abbildungen gleich groß.
- Einheitlichkeit > Kreativität: gleiche Breitenklassen, Schriftgrößen, Strichstärken, Tick- und Markerlogik in allen Figuren.
- Alle Beschriftungen in Englisch (kein „Wasser“, „Mittelwert“, „Partikelgroesse“).
- Keine Titel in der Figur (`ax.set_title` nur für Panel-Überschriften wie „35 nm“); die Aussage steht in der Caption.

## 1) Schriftarten

- **In allen Figuren: Open Sans** (serifenlos; Fallback Arial → DejaVu Sans). Gewichte nur Regular und Semibold/Bold.
- **Fließtext und Captions (LaTeX):** aktuell Latin Modern (`lmodern`).
  Falls Palatino gewünscht ist, in `Settings.tex` `\usepackage{newpxtext,newpxmath}` statt `lmodern` laden – dann gilt: Palatino nur im Fließtext, nie in Figuren.
- Formeln in Figuren (mathtext) ebenfalls in Open Sans setzen (siehe rcParams), damit $D_0$, $\tau$, µm² nicht in einer anderen Schrift erscheinen.

## 2) Figure-Format und Breitenklassen

Textbreite der Dissertation: **16,0 cm = 6,30 in**. Seitenverhältnis Standard **1,42 : 1** (width : height).

| Klasse | LaTeX-Einbindung | Breite | Höhe (1,42 : 1) |
|---|---|---|---|
| `full` | `\includegraphics[width=\linewidth]` | 6,30 in (16,0 cm) | 4,44 in |
| `narrow` | `\includegraphics[width=0.75\linewidth]` | 4,72 in (12,0 cm) | 3,32 in |
| `half` | `subfigure 0.49\linewidth` + `width=\linewidth` | 3,07 in (7,8 cm) | 2,16 in |
| `third` | `subfigure 0.32\linewidth` + `width=\linewidth` | 2,01 in (5,1 cm) | 1,42 in |

- Nur diese vier Breiten verwenden; Breite im Skript und in LaTeX gehören immer als Paar zusammen.
- Mehrpanel-Plots mit gemeinsamen Achsen (z. B. 2×3 nach Partikelgröße): **in Python als eine Figur** in Klasse `full` erzeugen (Höhe darf dann vom 1,42-Verhältnis abweichen).
- Unabhängige Einzelbilder (z. B. Mikroskopie-Frames, eMSD pro Messung): einzeln in `half`/`third` erzeugen und in LaTeX als `subfigure` anordnen.

## 3) Export

- Speicherort für alle Abbildungen und zugehörigen Ergebnis-Workbooks: `E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data`. Nicht im GitHub-Repository und nicht im Rohdatenordner speichern.

- **Diagramme (Plots): PDF** (Vektor) – scharf bei jedem Zoom, echte Schrift, kleine Dateien.
- **Mikroskopiebilder und Bildausschnitte: PNG, 600 dpi.**
- **Plots mit sehr vielen Elementen** (tausende Trajektorien, dichte Scatter): PDF mit gerasterten Datenebenen, z. B. `ax.plot(..., rasterized=True)`; Achsen und Text bleiben Vektor (`savefig.dpi = 600` gilt für die gerasterten Teile).
- **Kein `bbox_inches="tight"`** – das verändert die Bildgröße und damit die Schriftgröße nach dem Einbinden. Ränder stattdessen mit `layout="constrained"` (oder `tight_layout()`) innerhalb der festen Figurgröße regeln.
- Dateiname sprechend und ohne Leerzeichen/Umlaute, z. B. `emsd_hyd_35nm_surface.pdf`.
- Optional: Parameter und Ergebnisse als Sidecar-JSON gleichen Namens speichern (`emsd_hyd_35nm_surface.json`).

## 4) Typografie (Größen im fertigen PDF)

Bezug: Fließtext 11 pt, Captions 10 pt → Figurtext etwas kleiner.

| Element | Größe | Gewicht |
|---|---|---|
| Achsenbeschriftung | 9 pt | Regular |
| Ticklabels | 8 pt | Regular |
| Legende | 8 pt | Regular |
| Panel-Überschrift (z. B. „35 nm“) | 9 pt | Semibold |
| Panel-Labels (A, B, C …) | 10 pt | Bold |
| Colorbar-Label / -Ticks | 9 pt / 8 pt | Regular |
| Scalebar-Text (Mikroskopie) | 8 pt | Regular |
| Annotationen im Plot | 8 pt | Regular |

Regeln:
- Keine Kursivschrift außer Variablensymbolen ($D$, $n$, $\tau$).
- Einheiten konsequent mit Leerzeichen: 10 µm, 200 nm; Einheiten in Klammern: `Diffusion coefficient D (µm² s⁻¹)`, `Lag time τ (s)`.
- Keine All-Caps.

## 5) Achsen, Ticks, Spines

- Ticks auf allen Seiten (`top=True`, `right=True`), Richtung nach innen.
- Major-Ticks: Länge 3,5 pt, Breite 0,8 pt; Minor-Ticks: Länge 2,0 pt, Breite 0,6 pt.
- Spines: 0,8 pt.
- Grid: aus. Nur bei Bedarf sehr dezent (linewidth 0,5, alpha 0,2).
- Log-Achsen mit Minor-Ticks (`_add_log_minor_ticks()`).

## 6) Linien, Marker, Transparenz

Werte gelten im fertigen PDF (für alle Breitenklassen gleich).

- Datenlinie: 1,2 pt; bei mehreren Experimenten 1,0 pt.
- Fit-/Modelllinie: 1,5 pt, opak, Farbe `dark` der Serie (klar über der Datenlinie).
- Theorielinie (z. B. Stokes–Einstein): schwarz, 1,2 pt, gestrichelt `(0, (4, 3))`.
- Marker nur, wenn Messpunkte relevant sind:
  - `o` Hauptdaten, `s` Vergleichsdaten / Kategorie B, `^` dritte Gruppe
  - Markersize 4 pt (Scatter `s ≈ 16`), markeredgewidth 0,6
  - gefüllt; bei Überlagerung alpha 0,5–0,7
- Keine Transparenz bei Fits.

## 7) Scatter + Errorbars

- Errorbars: elinewidth 0,8, capsize 2,0, capthick 0,8.
- zorder: Errorbars 2, Marker 3.
- Viele Punkte: Marker alpha 0,6, Errorbars alpha 0,5.
- Punkte exakt auf dem realen x-Wert (kein Jitter); Häufungen werden durch Transparenz sichtbar.

## 8) Histogramme

- Bins 25–60; Form sichtbar, nicht kammartig. Gleiche Bin-Grenzen für verglichene Verteilungen.
- `stepfilled` oder `bar` mit dezenter Kante (0,5 pt), alpha 0,6 bei Überlagerung.
- KDE-/Fit-Kurve: 1,5 pt in der `dark`-Variante derselben Farbe.
- Maximal 2–3 Überlagerungen, sonst Panels.
- Verteilungsform in der Caption korrekt benennen (z. B. Schrittweite: rechtsschief/Rayleigh-artig; Tracklänge: annähernd exponentiell; nicht pauschal „Gaussian“).

## 9) Mikroskopiebilder

- Alle Bilder einer Abbildung: **gleiches Seitenverhältnis (1,42 : 1) und gleicher Maßstab**, wenn möglich gleiches Sichtfeld (z. B. 40 × 28 µm). Zuschneiden so, dass die Scalebar (unten rechts) erhalten bleibt.
- Kontrast-Darstellung dokumentieren (z. B. Perzentile 0,5–99,9, Gamma 0,6).
- Scalebar: einfarbiger Balken (weiß/schwarz je nach Hintergrund), Höhe ca. 1–1,5 % der Bildhöhe; Text 8 pt Regular oberhalb oder rechts, kleiner Abstand; bei unruhigem Hintergrund dünne Text-Outline statt fetter Schrift. Einheitliche Balkenlänge innerhalb einer Abbildung (z. B. überall 5 µm oder überall 10 µm).
- Annotationen: weiß/schwarz oder eine Akzentfarbe; Linienbreite 1,2 pt; Pfeile `arrowstyle='-|>'`, `mutation_scale` 10–12; ROI-Boxen 1,0 pt.
- Detektionskreise/Trajektorien: gleiche Farben in allen Frames (z. B. Kreise #da00bd, Trajektorien #004cff).

## 10) Colormap (fix): Jet für Geschwindigkeit/Diffusion vs. Partikelgröße

- `cmap = "jet"`, Bereich log 0 bis log 15, entsprechend Partikeln 20–1000 nm (nominal).
- Colorbar-Label z. B. `D (µm² s⁻¹, log scale)`; wenige Ticks (z. B. 0, 3, 6, 9, 12, 15); Ticklabels 8 pt.
- Hinweis: Jet ist nicht wahrnehmungsgleichmäßig; nur für diese festgelegte Darstellung verwenden, sonst `viridis`/`cividis`.

## 11) Farben

### Diskrete Serien (Jet-angelehnt)

| Index | base | dark | bright |
|---|---|---|---|
| 1 | #0000da | #000099 | #1f1fde |
| 2 | #004cff | #0035b2 | #1f61ff |
| 3 | #00c4ff | #0089b2 | #1fcbff |
| 4 | #49ffad | #33b279 | #5bffb8 |
| 5 | #adff49 | #79b233 | #b8ff5b |
| 6 | #ffd700 | #b29600 | #ffde1f |
| 7 | #ff6800 | #b24900 | #ff751f |
| 8 | #da0000 | #990000 | #de1f1f |

Besondere Markierungen: A #da00bd / #551344 / #9b5191 · B #512279 / #4f1055 / #9850b4 · C #237735 / #1a5e3c / #54ad49

### Gedämpfte Vergleichsfarben (Standard für Kategorienvergleiche)

| Rolle | base | dark |
|---|---|---|
| Kategorie A (z. B. Surface loading, Buffer) | #3B8C8C | #2A6666 |
| Kategorie B (z. B. Injection, Hydrogel) | #D98C3D | #A6672D |
| Einzelserie (z. B. D₀) | #3B6E8C | #2A4F66 |
| DLS-Referenz | #da00bd | #9b5191 |

### Partikelgrößen (feste Zuordnung)

Jede Partikelgröße hat in allen Figuren dieselbe Farbe. Richtung wie die Jet-Colormap in Abschnitt 10: kleine (schnelle) Partikel warm, große (langsame) Partikel blau.

| Nominal | Label (DLS) | base (Fläche/Marker) | dark (Linie/Kante) |
|---|---|---|---|
| 20 nm | 35 nm | #ff6800 | #b24900 |
| 50 nm | 50 nm | #ffd700 | #b29600 |
| 100 nm | 100 nm | #49ffad | #33b279 |
| 200 nm | 240 nm | #00c4ff | #0089b2 |
| 500 nm | 560 nm | #004cff | #0035b2 |
| 1000 nm | 1370 nm | #0000da | #000099 |

- Quelle im Code: `SIZE_COLORS` in `hydro_analysis/MSD_Trackmate/Validation_Claude/Correlations.py` als `{nominal_nm: (base, dark)}` – importieren, nicht kopieren.
- `base` für Markerflächen, gefüllte Flächen und Bänder; `dark` für Linien, Markerkanten und Errorbars. Dünne Linien immer in `dark`, da `base` für 100 nm (#49ffad) und 50 nm (#ffd700) auf Weiß kaum lesbar ist.
- Legendenbeschriftung immer mit dem DLS-Label (`get_dls_labels()`), nicht mit der Nominalgröße.
- Noch abweichend (vereinheitlichen): `Validation_Claude/Fittingpoints_robustness.py` (umgekehrte Reihenfolge, 20 nm blau) und `MSD_Trackmate/plot_emsd_publication.py` (Okabe-Ito-Farben).

Regeln:
- Datenserie `base`, Fit/Modell `dark`, Hervorhebung `bright` sparsam.
- **Feste Zuordnung über die ganze Arbeit:** Buffer/Water immer dieselbe Farbe, Hydrogel immer dieselbe, Surface loading / Injection immer dieselbe (derzeit z. B. Water grün vs. blau in verschiedenen Plots – vereinheitlichen).

## 12) Mehrere Experimente

- Maximal 3–4 Serien pro Achse, sonst Panels.
- Legende kurz, ohne Rahmen, verdeckt keine Daten (bei fallender Theoriekurve meist „upper right“).
- Hierarchie: Primärdaten kontrastreich, Referenz dünner / alpha 0,7.
- Gleiche Achsenskalierung bei Vergleichspanels (`sharex`, `sharey`).
- Panel-Labels (A, B, C …) links oben, 10 pt bold – oder in LaTeX über `subfigure` (dann nicht zusätzlich im Bild).

## 13) Dataset-Vergleichsplots (Messgröße vs. Partikelgröße, log-log)

Referenzimplementierung: `MSD_Diffusion_vs_Size_20mg.py` / `MSD_Diffusion_vs_Size_D0.py` (`hydro_analysis/MSD_Trackmate/`).

- Klasse `narrow` oder `full`, kein Titel.
- x-Achse: immer reale DLS-Größe (`get_dls_sizes()`, z-Average, Fallback `size_override_nm`), Ticks per FixedLocator/FixedFormatter mit `get_dls_labels()` (35/50/100/240/560/1370). Log-log mit Minor-Ticks.
- Messpunkte exakt am realen x-Wert, alpha ≈ 0,5; Errorbars wie Abschnitt 7.
- Theorie-Linie wie Abschnitt 6.
- DLS-Referenz: Quadrat, markersize 5, opak, #da00bd / #9b5191, x- und y-Fehlerbalken.
- Kategorien: A `o` #3B8C8C, B `s` #D98C3D (Abschnitt 11).

## 14) Matplotlib-Vorlage

`thesis.mplstyle` (abgelegt unter `hydro_analysis/thesis.mplstyle`, in jedem Skript per Pfad laden, z. B. `plt.style.use(Path(__file__).resolve().parents[1] / "thesis.mplstyle")`):

```ini
figure.dpi          : 150
savefig.dpi         : 600
savefig.format      : pdf
savefig.bbox        : standard
savefig.pad_inches  : 0.02
figure.constrained_layout.use : True

font.family         : sans-serif
font.sans-serif     : Open Sans, Arial, DejaVu Sans
font.size           : 9
axes.labelsize      : 9
axes.titlesize      : 9
axes.titleweight    : semibold
xtick.labelsize     : 8
ytick.labelsize     : 8
legend.fontsize     : 8
legend.frameon      : False
mathtext.fontset    : custom
mathtext.rm         : Open Sans
mathtext.it         : Open Sans:italic
mathtext.bf         : Open Sans:bold

axes.linewidth      : 0.8
axes.grid           : False
xtick.direction     : in
ytick.direction     : in
xtick.top           : True
ytick.right         : True
xtick.major.size    : 3.5
ytick.major.size    : 3.5
xtick.minor.size    : 2.0
ytick.minor.size    : 2.0
xtick.major.width   : 0.8
ytick.major.width   : 0.8
xtick.minor.width   : 0.6
ytick.minor.width   : 0.6

lines.linewidth     : 1.2
lines.markersize    : 4
lines.markeredgewidth : 0.6
errorbar.capsize    : 2
```

Hilfsfunktionen:

```python
import matplotlib.pyplot as plt

WIDTH_IN = {"full": 6.30, "narrow": 4.72, "half": 3.07, "third": 2.01}
RATIO = 1.42
DASH_THEORY = (0, (4, 3))
ARROW_KW = dict(arrowstyle="-|>", mutation_scale=11, lw=1.2)

def new_figure(kind="half", nrows=1, ncols=1, height=None, **kw):
    """Figure in der exakten Endgröße für die Dissertation."""
    w = WIDTH_IN[kind]
    h = height if height is not None else w / RATIO
    return plt.subplots(nrows, ncols, figsize=(w, h), **kw)

def save_figure(fig, path, raster=False):
    """Plots als PDF, Mikroskopie (raster=True) als PNG 600 dpi. Kein bbox='tight'."""
    fmt = "png" if raster else "pdf"
    fig.savefig(f"{path}.{fmt}", format=fmt, dpi=600)
```

Beispiel:

```python
plt.style.use("thesis.mplstyle")
fig, ax = new_figure("half")            # -> subfigure 0.49\linewidth
ax.loglog(tau, emsd, "o-", color="#3B8C8C")
ax.loglog(tau, fit, color="#2A6666", lw=1.5)
ax.loglog(tau, se, color="k", lw=1.2, ls=DASH_THEORY)
ax.set_xlabel(r"Lag time $\tau$ (s)")
ax.set_ylabel(r"MSD (µm$^2$)")
save_figure(fig, "emsd_hyd_35nm_surface")
```

## 15) Einbindung in LaTeX

```latex
% full
\includegraphics[width=\linewidth]{Pictures/Data/SPT/diffusion_vs_size_d0.pdf}
% narrow
\includegraphics[width=0.75\linewidth]{...}
% half / third
\begin{subfigure}[t]{0.49\linewidth}\centering
  \includegraphics[width=\linewidth]{...}
\end{subfigure}
```

Keine anderen Breiten (z. B. 0.8 oder 0.9\linewidth) – sonst stimmt die Schriftgröße nicht mehr.
