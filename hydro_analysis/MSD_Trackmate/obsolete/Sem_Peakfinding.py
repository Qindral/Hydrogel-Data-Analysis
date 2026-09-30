import numpy as np
import matplotlib.pyplot as plt
import json

from pathlib import Path
from PIL import Image

from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import remove_small_objects, closing, disk
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from scipy import ndimage as ndi
import pandas as pd

from hydro_analysis.core.io import extract_particle_size_from_path

# falls du die Pixelgröße aus den FEI-Metadaten holen willst:
# from read_fei_sem_metadata import summarize_sem_metadata

# ============================================================================
# PARTIKEL-GRÖSSENPARAMETER (in Nanometern - werden automatisch in Pixel umgerechnet)
# ============================================================================


# ============================================================================
# FILTERPARAMETER
# ============================================================================
# GAUSSIAN_SIGMA wird basierend auf Pixelgröße automatisch angepasst (siehe Funktion)
# Manuelles Override möglich, wenn nicht None:
GAUSSIAN_SIGMA_OVERRIDE = None  # Setze auf z.B. 7 für festen Wert, None für automatisch

MIN_CIRCULARITY = 0.22
MAX_CIRCULARITY = 1
MIN_INTENSITYMAX = 0.0  # max Intensität mindestens

# Qualitätsfilter für Partikel-Fitting
MIN_QUALITY_SCORE = 0.01  # Minimale Qualität des Profils (0-1)
MIN_CENTER_QUALITY = 0.2  # Minimale Zentrierungsqualität (0-1)
# MAX_CENTER_OFFSET_PX wird basierend auf Pixelgröße automatisch angepasst
MAX_CENTER_OFFSET_NM = 20.0  # Maximale Abweichung des Zentrums in nm (wird zu Pixel umgerechnet)
MAX_RADIUS_CHANGE_FACTOR = 0.4  # Maximale relative Änderung des Radius (z.B. 0.7 = 70%)

# Faktoren für Bereichsberechnung (können angepasst werden)
MIN_AREA_FACTOR = .85  # Faktor für minimale Fläche: factor * π * (min_radius)^2
MAX_AREA_FACTOR = 1.1  # Faktor für maximale Fläche: factor * π * (max_radius)^2

# Debug-Modus: Zeigt an, warum Partikel herausgefiltert werden
DEBUG_FILTERING = False

OUTPUT_CSV = "segmented_particles_watershed.csv"

# Threshold-Einstellungen Datei
THRESHOLD_CONFIG_FILE = "particle_thresholds.json"

# Ausgabe-Verzeichnis für Auswertungsdateien
OUTPUT_DIR = Path(r"D:\SEM\Auswertung")

# ============================================================================
# THRESHOLD SPEICHERN / LADEN
# ============================================================================

def save_thresholds(contrast_thresh, uniformity_thresh, filepath=None):
    """
    Speichert die gewählten Threshold-Werte in eine JSON-Datei.
    """
    if filepath is None:
        filepath = THRESHOLD_CONFIG_FILE

    config = {
        'contrast_threshold': float(contrast_thresh),
        'uniformity_threshold': float(uniformity_thresh),
    }

    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Thresholds gespeichert: {filepath}")
    return config


def load_thresholds(filepath=None):
    """
    Lädt gespeicherte Threshold-Werte aus einer JSON-Datei.
    Gibt Default-Werte zurück falls Datei nicht existiert.
    """
    if filepath is None:
        filepath = THRESHOLD_CONFIG_FILE

    default_config = {
        'contrast_threshold': 0.3,
        'uniformity_threshold': 0.3,
    }

    try:
        with open(filepath, 'r') as f:
            config = json.load(f)
        print(f"Thresholds geladen: Kontrast={config['contrast_threshold']:.2f}, "
              f"Uniformität={config['uniformity_threshold']:.2f}")
        return config
    except FileNotFoundError:
        print(f"Keine gespeicherten Thresholds gefunden, verwende Defaults.")
        return default_config


def add_scale_bar(ax, px_nm, image_shape, bar_length_nm=None, location='lower right',
                  color='white', fontsize=10, box_alpha=0.7):
    """
    Fügt eine Maßstabsleiste (Scale Bar) zu einem Matplotlib-Axes hinzu.

    Args:
        ax: Matplotlib Axes Objekt
        px_nm: Pixelgröße in nm/pixel
        image_shape: (height, width) des Bildes in Pixel
        bar_length_nm: Länge der Scale Bar in nm (None = automatisch)
        location: Position ('lower right', 'lower left', 'upper right', 'upper left')
        color: Farbe der Scale Bar und des Texts
        fontsize: Schriftgröße
        box_alpha: Transparenz der Hintergrundbox
    """
    height, width = image_shape[:2]

    # Automatische Länge wählen (ca. 10-20% der Bildbreite)
    if bar_length_nm is None:
        image_width_nm = width * px_nm
        # Wähle eine "schöne" Länge (100, 200, 500, 1000, 2000, 5000 nm etc.)
        nice_lengths = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        target_length = image_width_nm * 0.15  # ca. 15% der Bildbreite
        bar_length_nm = min(nice_lengths, key=lambda x: abs(x - target_length))

    bar_length_px = bar_length_nm / px_nm

    # Position berechnen
    margin = 0.05  # 5% Rand
    bar_height_px = max(5, height * 0.01)  # Höhe der Bar

    if 'right' in location:
        x_start = width * (1 - margin) - bar_length_px
    else:
        x_start = width * margin

    if 'lower' in location:
        y_pos = height * (1 - margin)
    else:
        y_pos = height * margin

    # Scale Bar zeichnen
    from matplotlib.patches import Rectangle, FancyBboxPatch

    # Hintergrund-Box
    box_padding = 10
    box_width = bar_length_px + 2 * box_padding
    box_height = bar_height_px + fontsize * 2 + box_padding

    if 'lower' in location:
        box_y = y_pos - box_height
    else:
        box_y = y_pos

    if 'right' in location:
        box_x = x_start - box_padding
    else:
        box_x = x_start - box_padding

    bg_box = FancyBboxPatch((box_x, box_y), box_width, box_height,
                             boxstyle="round,pad=3", facecolor='black',
                             alpha=box_alpha, edgecolor='none', zorder=10)
    ax.add_patch(bg_box)

    # Scale Bar selbst
    bar = Rectangle((x_start, y_pos - bar_height_px), bar_length_px, bar_height_px,
                    facecolor=color, edgecolor=color, zorder=11)
    ax.add_patch(bar)

    # Label
    if bar_length_nm >= 1000:
        label = f"{bar_length_nm/1000:.0f} µm"
    else:
        label = f"{bar_length_nm:.0f} nm"

    ax.text(x_start + bar_length_px / 2, y_pos - bar_height_px - 5, label,
            ha='center', va='bottom', color=color, fontsize=fontsize,
            fontweight='bold', zorder=12)


def save_segmented_image(image, particles, px_nm, output_path, title=None,
                         show_accepted_only=False, thresholds=None):
    """
    Speichert ein Bild mit eingezeichneten Partikeln und Scale Bar.

    Args:
        image: Graustufenbild (numpy array)
        particles: Liste von Partikel-Dicts
        px_nm: Pixelgröße in nm/pixel
        output_path: Pfad zum Speichern
        title: Optionaler Titel
        show_accepted_only: Wenn True, nur akzeptierte Partikel zeigen
        thresholds: Dict mit 'contrast' und 'uniformity' Thresholds
    """
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(image, cmap='gray', vmin=0, vmax=255 if image.max() > 1 else 1)

    cmap = plt.cm.RdYlGn
    norm = Normalize(vmin=0, vmax=1)

    # Partikel einzeichnen
    for p in particles:
        # Prüfe ob Partikel akzeptiert ist (falls Thresholds gegeben)
        if thresholds:
            c_score = p.get('contrast_score', 0)
            u_score = p.get('uniformity_score', 0)
            is_accepted = (c_score >= thresholds['contrast'] and
                          u_score >= thresholds['uniformity'])
            if show_accepted_only and not is_accepted:
                continue
            alpha = 1.0 if is_accepted else 0.3
            linestyle = '-' if is_accepted else ':'
        else:
            alpha = 1.0
            linestyle = '-'

        cx, cy = p['x_px'], p['y_px']
        r = p['refined_radius_px']
        contrast = p.get('contrast_score', 0.5)
        uniformity = p.get('uniformity_score', 0.5)
        color = cmap(norm(contrast))
        linewidth = 0.5 + 3.0 * uniformity

        circle = plt.Circle((cx, cy), r, fill=False, color=color,
                            linewidth=linewidth, alpha=alpha, linestyle=linestyle)
        ax.add_patch(circle)

    # Scale Bar hinzufügen
    add_scale_bar(ax, px_nm, image.shape)

    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.03, pad=0.01)
    cbar.set_label('Kontrast-Score')

    # Titel
    if title:
        ax.set_title(title, fontsize=12)

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Segmentiertes Bild gespeichert: {output_path}")


def filter_particles_by_thresholds(particles, contrast_thresh=None, uniformity_thresh=None):
    """
    Filtert Partikel basierend auf Threshold-Werten.

    Wenn keine Thresholds angegeben werden, werden die gespeicherten Werte verwendet.

    Args:
        particles: Liste von Partikel-Dicts mit 'contrast_score' und 'uniformity_score'
        contrast_thresh: Kontrast-Score Threshold (None = aus Datei laden)
        uniformity_thresh: Uniformitäts-Score Threshold (None = aus Datei laden)

    Returns:
        Liste der akzeptierten Partikel
    """
    # Thresholds laden falls nicht angegeben
    if contrast_thresh is None or uniformity_thresh is None:
        config = load_thresholds()
        if contrast_thresh is None:
            contrast_thresh = config['contrast_threshold']
        if uniformity_thresh is None:
            uniformity_thresh = config['uniformity_threshold']

    # Partikel filtern
    accepted = [p for p in particles
                if p.get('contrast_score', 0) >= contrast_thresh
                and p.get('uniformity_score', 0) >= uniformity_thresh]

    print(f"Filterung: {len(accepted)}/{len(particles)} Partikel akzeptiert "
          f"(Kontrast≥{contrast_thresh:.2f}, Uniformität≥{uniformity_thresh:.2f})")

    return accepted


# ============================================================================
# PARTIKEL-VALIDIERUNG UND QUALITÄTSBEWERTUNG
# ============================================================================

def validate_particle(image, cx, cy, r, ring_width=5, n_theta=64):
    """
    Bewertet einen detektierten Partikel mit ZWEI kontinuierlichen Metriken:

    1. KONTRAST-SCORE (für Farbe):
       - Misst wie gut sich der Partikel vom Hintergrund abhebt
       - Hoher Wert = Kante liegt richtig, guter Kontrast

    2. UNIFORMITÄTS-SCORE (für Liniendicke):
       - Misst die radiale Gleichmäßigkeit im Inneren
       - Hoher Wert = gleichmäßig, keine internen Kanten

    Args:
        image: Graustufenbild (float)
        cx, cy: Zentrum des Partikels
        r: Radius des Partikels
        ring_width: Breite des Außenrings für Vergleich (Pixel)
        n_theta: Anzahl Abtastpunkte pro Kreis

    Returns:
        dict mit beiden Scores und Details
    """
    # Sicherstellen, dass wir innerhalb des Bildes bleiben
    if (cx < r + ring_width or cx > image.shape[1] - r - ring_width or
        cy < r + ring_width or cy > image.shape[0] - r - ring_width):
        return {
            'contrast_score': 0.0,
            'uniformity_score': 0.0,
            'quality_score': 0.0,
            'details': {}
        }

    # --- 1. Intensitäten samplen ---
    # Innenbereich (mehrere Radien)
    inner_intensities = []
    for r_sample in np.linspace(0.2 * r, 0.8 * r, 5):
        if r_sample > 2:
            vals = sample_circle(image, cx, cy, r_sample, n_theta)
            inner_intensities.extend(vals)

    # Auf dem Rand (Kante)
    edge_intensities = sample_circle(image, cx, cy, r, n_theta)

    # Außenring (5 Pixel außerhalb)
    outer_intensities = []
    for r_sample in np.linspace(r + 2, r + ring_width, 3):
        vals = sample_circle(image, cx, cy, r_sample, n_theta)
        outer_intensities.extend(vals)

    inner_intensities = np.array(inner_intensities)
    edge_intensities = np.array(edge_intensities)
    outer_intensities = np.array(outer_intensities)

    # --- 2. Metriken berechnen ---
    inner_mean = np.mean(inner_intensities)
    inner_std = np.std(inner_intensities)
    edge_mean = np.mean(edge_intensities)
    edge_std = np.std(edge_intensities)
    outer_mean = np.mean(outer_intensities)

    # === KONTRAST-SCORE (0-1) ===
    # Kombiniert: Kontrast Innen/Außen + Kantenschärfe + Kanten-Gleichmäßigkeit
    contrast = abs(inner_mean - outer_mean)
    contrast_normalized = contrast / (max(inner_mean, outer_mean) + 1e-9)

    # Kantenschärfe: Kante sollte zwischen Innen und Außen liegen
    edge_position_quality = 1.0 - abs(edge_mean - (inner_mean + outer_mean) / 2) / (contrast + 1e-9)
    edge_position_quality = max(0.0, min(1.0, edge_position_quality))

    # Kanten-Gleichmäßigkeit (niedrige Varianz = gute Kante)
    edge_uniformity = 1.0 / (1.0 + edge_std / (abs(inner_mean - outer_mean) + 1e-9))

    contrast_score = (
        0.4 * min(contrast_normalized * 3, 1.0) +  # Kontrast (verstärkt)
        0.3 * edge_position_quality +               # Kantenposition
        0.3 * edge_uniformity                       # Kanten-Gleichmäßigkeit
    )
    contrast_score = max(0.0, min(1.0, contrast_score))

    # === UNIFORMITÄTS-SCORE (0-1) ===
    # Misst wie gleichmäßig das Innere ist (keine internen Kanten)
    radial_uniformity = 1.0 / (1.0 + inner_std / (inner_mean + 1e-9) * 5)

    # Prüfe auf interne Gradienten (sollten klein sein)
    # Sample auf verschiedenen Radien und prüfe Konsistenz
    radial_means = []
    for r_sample in np.linspace(0.3 * r, 0.7 * r, 4):
        if r_sample > 2:
            vals = sample_circle(image, cx, cy, r_sample, n_theta)
            radial_means.append(np.mean(vals))

    if len(radial_means) > 1:
        radial_gradient = np.std(radial_means) / (np.mean(radial_means) + 1e-9)
        internal_smoothness = 1.0 / (1.0 + radial_gradient * 10)
    else:
        internal_smoothness = 0.5

    uniformity_score = (
        0.6 * radial_uniformity +
        0.4 * internal_smoothness
    )
    uniformity_score = max(0.0, min(1.0, uniformity_score))

    # Gesamtqualität (Durchschnitt beider Scores)
    quality_score = (contrast_score + uniformity_score) / 2

    return {
        'contrast_score': float(contrast_score),
        'uniformity_score': float(uniformity_score),
        'quality_score': float(quality_score),
        'details': {
            'inner_mean': float(inner_mean),
            'inner_std': float(inner_std),
            'edge_mean': float(edge_mean),
            'edge_std': float(edge_std),
            'outer_mean': float(outer_mean),
            'contrast': float(contrast),
            'contrast_normalized': float(contrast_normalized),
            'edge_position_quality': float(edge_position_quality),
            'edge_uniformity': float(edge_uniformity),
            'radial_uniformity': float(radial_uniformity),
            'internal_smoothness': float(internal_smoothness),
        }
    }


def refine_particle_position(image, cx, cy, r, max_shift=None, n_steps=5):
    """
    Versucht die Partikelposition zu optimieren.

    Sucht in einem Gitter um die aktuelle Position nach besserer Platzierung.

    Args:
        image: Graustufenbild
        cx, cy: Aktuelle Zentrumsposition
        r: Radius
        max_shift: Maximale Verschiebung (Standard: 30% des Radius)
        n_steps: Anzahl Schritte pro Richtung

    Returns:
        cx_new, cy_new, quality_improved
    """
    if max_shift is None:
        max_shift = r * 0.3

    best_cx, best_cy = cx, cy
    best_quality = validate_particle(image, cx, cy, r)['quality_score']

    # Grid-Suche
    for dx in np.linspace(-max_shift, max_shift, n_steps):
        for dy in np.linspace(-max_shift, max_shift, n_steps):
            cx_try = cx + dx
            cy_try = cy + dy

            validation = validate_particle(image, cx_try, cy_try, r)
            if validation['quality_score'] > best_quality:
                best_quality = validation['quality_score']
                best_cx = cx_try
                best_cy = cy_try

    return best_cx, best_cy, best_quality > validate_particle(image, cx, cy, r)['quality_score']


def visualize_validation_results(image, particles, save_path=None):
    """
    Visualisiert die Validierungsergebnisse mit ZWEI Metriken (statisch).
    Für interaktive Version siehe: interactive_threshold_viewer()
    """
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- 1. Bild mit farbcodierten Kreisen ---
    axes[0].imshow(image, cmap='gray')

    cmap = plt.cm.RdYlGn
    norm = Normalize(vmin=0, vmax=1)

    for p in particles:
        cx, cy = p['x_px'], p['y_px']
        r = p['refined_radius_px']
        contrast = p.get('contrast_score', 0.5)
        uniformity = p.get('uniformity_score', 0.5)
        color = cmap(norm(contrast))
        linewidth = 0.5 + 3.0 * uniformity
        circle = plt.Circle((cx, cy), r, fill=False, color=color, linewidth=linewidth)
        axes[0].add_patch(circle)

    axes[0].set_title('Partikel-Bewertung\nFarbe = Kontrast, Dicke = Uniformität')
    axes[0].axis('off')

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[0], orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Kontrast-Score', fontsize=10)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', linewidth=0.5, label='Uniformität: 0.0'),
        Line2D([0], [0], color='gray', linewidth=2.0, label='Uniformität: 0.5'),
        Line2D([0], [0], color='gray', linewidth=3.5, label='Uniformität: 1.0'),
    ]
    axes[0].legend(handles=legend_elements, loc='upper right', fontsize=8)

    # --- 2. Histogramm Kontrast-Score ---
    contrast_scores = [p.get('contrast_score', 0) for p in particles]
    n_bins = 20
    counts, bins, patches = axes[1].hist(contrast_scores, bins=n_bins, edgecolor='black')
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    for patch, bc in zip(patches, bin_centers):
        patch.set_facecolor(cmap(norm(bc)))
    axes[1].axvline(np.mean(contrast_scores), color='black', linestyle='--', linewidth=2,
                   label=f'Mittel: {np.mean(contrast_scores):.2f}')
    axes[1].set_xlabel('Kontrast-Score', fontsize=11)
    axes[1].set_ylabel('Anzahl Partikel', fontsize=11)
    axes[1].set_title('Verteilung Kontrast-Score')
    axes[1].legend()

    # --- 3. Histogramm Uniformitäts-Score ---
    uniformity_scores = [p.get('uniformity_score', 0) for p in particles]
    counts2, bins2, patches2 = axes[2].hist(uniformity_scores, bins=n_bins,
                                            edgecolor='black', color='steelblue', alpha=0.7)
    bin_centers2 = 0.5 * (bins2[:-1] + bins2[1:])
    for bc, count in zip(bin_centers2, counts2):
        if count > 0:
            linewidth = 0.5 + 3.0 * bc
            axes[2].plot([bc, bc], [0, -max(counts2) * 0.1], color='gray',
                        linewidth=linewidth, solid_capstyle='round')
    axes[2].axvline(np.mean(uniformity_scores), color='red', linestyle='--', linewidth=2,
                   label=f'Mittel: {np.mean(uniformity_scores):.2f}')
    axes[2].set_xlabel('Uniformitäts-Score', fontsize=11)
    axes[2].set_ylabel('Anzahl Partikel', fontsize=11)
    axes[2].set_title('Verteilung Uniformitäts-Score')
    axes[2].legend()
    axes[2].set_ylim(bottom=-max(counts2) * 0.15)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    # Statistik
    print(f"\n{'='*60}")
    print("BEWERTUNGSSTATISTIK:")
    print(f"{'='*60}")
    print(f"Gesamt: {len(particles)} Partikel")
    print(f"Kontrast-Score:    Mittel={np.mean(contrast_scores):.3f}, Std={np.std(contrast_scores):.3f}")
    print(f"Uniformitäts-Score: Mittel={np.mean(uniformity_scores):.3f}, Std={np.std(uniformity_scores):.3f}")
    print(f"{'='*60}\n")


def interactive_threshold_viewer(image, particles, load_previous=True):
    """
    INTERAKTIVE Visualisierung mit Slider für Threshold-Werte.

    Ermöglicht das Einstellen von:
    - Kontrast-Score Threshold
    - Uniformitäts-Score Threshold

    Zeigt in Echtzeit welche Partikel die Kriterien erfüllen.
    Speichert die gewählten Thresholds automatisch für spätere Verwendung.

    Args:
        image: Originalbild
        particles: Liste von dicts mit x_px, y_px, refined_radius_px, contrast_score, uniformity_score
        load_previous: Wenn True, werden vorherige Threshold-Werte als Startwerte geladen

    Returns:
        final_thresh_c, final_thresh_u, accepted_particles
    """
    from matplotlib.widgets import Slider, Button
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    # Lade vorherige Thresholds als Startwerte
    if load_previous:
        saved_config = load_thresholds()
        init_thresh_c = saved_config['contrast_threshold']
        init_thresh_u = saved_config['uniformity_threshold']
    else:
        init_thresh_c = 0.3
        init_thresh_u = 0.3

    # Daten extrahieren
    contrast_scores = np.array([p.get('contrast_score', 0) for p in particles])
    uniformity_scores = np.array([p.get('uniformity_score', 0) for p in particles])

    # Figure erstellen
    fig = plt.figure(figsize=(16, 10))

    # Layout: Bild links, Histogramme rechts, Slider unten
    ax_image = fig.add_axes([0.05, 0.25, 0.55, 0.70])
    ax_hist_contrast = fig.add_axes([0.65, 0.55, 0.30, 0.35])
    ax_hist_uniform = fig.add_axes([0.65, 0.12, 0.30, 0.35])

    # Slider-Achsen
    ax_slider_contrast = fig.add_axes([0.15, 0.12, 0.35, 0.03])
    ax_slider_uniform = fig.add_axes([0.15, 0.06, 0.35, 0.03])

    # Info-Text Achse
    ax_info = fig.add_axes([0.05, 0.01, 0.50, 0.04])
    ax_info.axis('off')

    # Colormap
    cmap = plt.cm.RdYlGn
    norm = Normalize(vmin=0, vmax=1)

    # --- Initiales Bild zeichnen ---
    ax_image.imshow(image, cmap='gray')
    ax_image.set_title('Partikel-Filterung (interaktiv)')
    ax_image.axis('off')

    # Kreise als Liste speichern für Updates
    circles = []
    for p in particles:
        cx, cy = p['x_px'], p['y_px']
        r = p['refined_radius_px']
        contrast = p.get('contrast_score', 0.5)
        uniformity = p.get('uniformity_score', 0.5)
        color = cmap(norm(contrast))
        linewidth = 0.5 + 3.0 * uniformity
        circle = plt.Circle((cx, cy), r, fill=False, color=color, linewidth=linewidth)
        ax_image.add_patch(circle)
        circles.append(circle)

    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_image, orientation='vertical', fraction=0.03, pad=0.01)
    cbar.set_label('Kontrast-Score')

    # --- Histogramme zeichnen ---
    # Kontrast-Histogramm
    n_bins = 20
    counts_c, bins_c, patches_c = ax_hist_contrast.hist(contrast_scores, bins=n_bins, edgecolor='black')
    bin_centers_c = 0.5 * (bins_c[:-1] + bins_c[1:])
    for patch, bc in zip(patches_c, bin_centers_c):
        patch.set_facecolor(cmap(norm(bc)))
    threshold_line_c = ax_hist_contrast.axvline(init_thresh_c, color='red', linestyle='-', linewidth=2, label='Threshold')
    ax_hist_contrast.set_xlabel('Kontrast-Score')
    ax_hist_contrast.set_ylabel('Anzahl')
    ax_hist_contrast.set_title('Kontrast-Score Verteilung')
    ax_hist_contrast.legend()

    # Uniformitäts-Histogramm
    counts_u, bins_u, patches_u = ax_hist_uniform.hist(uniformity_scores, bins=n_bins,
                                                        edgecolor='black', color='steelblue', alpha=0.7)
    threshold_line_u = ax_hist_uniform.axvline(init_thresh_u, color='red', linestyle='-', linewidth=2, label='Threshold')
    ax_hist_uniform.set_xlabel('Uniformitäts-Score')
    ax_hist_uniform.set_ylabel('Anzahl')
    ax_hist_uniform.set_title('Uniformitäts-Score Verteilung')
    ax_hist_uniform.legend()

    # --- Slider erstellen (mit geladenen Startwerten) ---
    slider_contrast = Slider(
        ax_slider_contrast, 'Kontrast\nThreshold',
        0.0, 1.0, valinit=init_thresh_c, valstep=0.01,
        color='green'
    )
    slider_uniform = Slider(
        ax_slider_uniform, 'Uniformität\nThreshold',
        0.0, 1.0, valinit=init_thresh_u, valstep=0.01,
        color='steelblue'
    )

    # Info-Text
    info_text = ax_info.text(0.5, 0.5, '', ha='center', va='center', fontsize=12,
                             transform=ax_info.transAxes)

    def update(val):
        """Update-Funktion für Slider"""
        thresh_c = slider_contrast.val
        thresh_u = slider_uniform.val

        # Zähler für akzeptierte Partikel
        n_accepted = 0
        accepted_diams = []

        # Kreise updaten
        for i, (circle, p) in enumerate(zip(circles, particles)):
            c_score = p.get('contrast_score', 0)
            u_score = p.get('uniformity_score', 0)

            if c_score >= thresh_c and u_score >= thresh_u:
                # Akzeptiert: normal anzeigen
                circle.set_alpha(1.0)
                circle.set_linestyle('-')
                n_accepted += 1
                accepted_diams.append(p['refined_diam_nm'])
            else:
                # Abgelehnt: transparent und gestrichelt
                circle.set_alpha(0.15)
                circle.set_linestyle(':')

        # Threshold-Linien updaten
        threshold_line_c.set_xdata([thresh_c, thresh_c])
        threshold_line_u.set_xdata([thresh_u, thresh_u])

        # Info-Text updaten
        if accepted_diams:
            mean_d = np.mean(accepted_diams)
            std_d = np.std(accepted_diams)
            info = (f"Akzeptiert: {n_accepted}/{len(particles)} Partikel ({100*n_accepted/len(particles):.1f}%)  |  "
                   f"Durchmesser: {mean_d:.1f} ± {std_d:.1f} nm")
        else:
            info = f"Akzeptiert: 0/{len(particles)} Partikel (0%)"
        info_text.set_text(info)

        fig.canvas.draw_idle()

    # Slider mit Update verbinden
    slider_contrast.on_changed(update)
    slider_uniform.on_changed(update)

    # Initial update
    update(None)

    plt.show()

    # Finale Werte zurückgeben
    final_thresh_c = slider_contrast.val
    final_thresh_u = slider_uniform.val

    # Thresholds für spätere Verwendung speichern
    save_thresholds(final_thresh_c, final_thresh_u)

    print(f"\n{'='*60}")
    print("GEWÄHLTE THRESHOLD-WERTE:")
    print(f"{'='*60}")
    print(f"Kontrast-Score Threshold:    {final_thresh_c:.2f}")
    print(f"Uniformitäts-Score Threshold: {final_thresh_u:.2f}")

    accepted = [p for p in particles
                if p.get('contrast_score', 0) >= final_thresh_c
                and p.get('uniformity_score', 0) >= final_thresh_u]
    print(f"\nAkzeptierte Partikel: {len(accepted)}/{len(particles)}")
    if accepted:
        diams = [p['refined_diam_nm'] for p in accepted]
        print(f"Durchmesser: {np.mean(diams):.1f} ± {np.std(diams):.1f} nm")
    print(f"{'='*60}\n")

    return final_thresh_c, final_thresh_u, accepted


def create_circular_template(radius, edge_width=2):
    """
    Erstellt ein kreisförmiges Template für Template Matching.

    Das Template hat:
    - Hohe Werte im Inneren
    - Fallende Werte am Rand (Kante)
    - Niedrige Werte außen
    """
    size = int(2 * radius + 10)
    if size % 2 == 0:
        size += 1  # Ungerade Größe für symmetrisches Template

    center = size // 2
    y, x = np.ogrid[:size, :size]
    dist = np.sqrt((x - center)**2 + (y - center)**2)

    # Kreisförmiges Template mit weichem Rand
    template = np.zeros((size, size), dtype=float)

    # Innerer Bereich: hoch
    inner_mask = dist < (radius - edge_width)
    template[inner_mask] = 1.0

    # Randbereich: Gradient
    edge_mask = (dist >= radius - edge_width) & (dist <= radius + edge_width)
    template[edge_mask] = 0.5 * (1 + np.cos(np.pi * (dist[edge_mask] - (radius - edge_width)) / (2 * edge_width)))

    # Normalisieren
    template = (template - template.mean()) / (template.std() + 1e-9)

    return template


def segment_particles_template_matching(tif_path, particle_nm, n_scales=14,
                                        size_factor_min=0.8, size_factor_max=1.2):
    """
    Partikelsegmentierung mit Multi-Scale Template Matching.

    Besonders gut für:
    - Partikel mit bekannter Größe
    - Überlappende Partikel
    - Konsistente Partikelformen

    Args:
        tif_path: Pfad zum TIFF-Bild
        particle_nm: Erwartete Partikelgröße in nm (aus Dateiname/Pfad extrahiert)
        n_scales: Anzahl der Skalen für Template Matching
        size_factor_min: Faktor für minimale Partikelgröße (default: 0.6)
        size_factor_max: Faktor für maximale Partikelgröße (default: 1.5)

    Ausgabedateien werden im selben Ordner wie die Eingabedatei gespeichert:
        - segmented_particles_of_{filename}.csv (alle Partikel)
        - segmented_particles_of_{filename}_accepted.csv (akzeptierte Partikel)
        - segmented_of_{filename}.png (Bild mit Segmentierung und Scalebar)
        - histogram_of_{filename}.png
        - hist_partikel_of_{filename}_um.txt
    """
    from skimage.feature import match_template

    tif_path = Path(tif_path)

    # --- Ausgabe-Verzeichnis und Basis-Dateiname ---
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = tif_path.stem  # Dateiname ohne Erweiterung

    # --- Erwartete Partikelgröße aus übergebenem Parameter ---
    expected_diam_min = particle_nm * size_factor_min
    expected_diam_max = particle_nm * size_factor_max

    # --- Pixelgröße aus FEI-Metadaten ---
    meta = summarize_sem_metadata(tif_path)
    px_nm = 0.5 * (meta["summary"]["pixel_width_nm"] + meta["summary"]["pixel_height_nm"])

    if px_nm is None or px_nm <= 0:
        raise ValueError("Pixelgröße konnte nicht aus Metadaten extrahiert werden.")

    print(f"\n{'='*60}")
    print(f"TEMPLATE MATCHING SEGMENTIERUNG")
    print(f"{'='*60}")
    print(f"Datei: {tif_path.name}")
    print(f"Pixelgröße: {px_nm:.4f} nm/pixel")
    print(f"Erwartete Partikelgröße: {expected_diam_min:.0f} - {expected_diam_max:.0f} nm (Nominal: {particle_nm} nm)")

    # --- Radius in Pixel ---
    min_radius_px = (expected_diam_min / 2.0) / px_nm
    max_radius_px = (expected_diam_max / 2.0) / px_nm

    print(f"  Min. Radius: {min_radius_px:.2f} px, Max. Radius: {max_radius_px:.2f} px")
    print(f"{'='*60}\n")

    # --- Bild laden ---
    pil_img = Image.open(tif_path)
    if np.max(pil_img) < 255:
        img_gray = pil_img.convert("L")
    else:
        img_gray = pil_img
    image_original = np.array(img_gray)  # Originalbild für Anzeige (uint8)
    image = np.array(img_gray, dtype=float)  # Float für Berechnungen

    # Normalisieren NUR für Template Matching (nicht für Anzeige)
    image_norm = (image - image.mean()) / (image.std() + 1e-9)

    # --- Multi-Scale Template Matching ---
    radii = np.linspace(min_radius_px, max_radius_px, n_scales)
    all_detections = []

    for r in radii:
        template = create_circular_template(r)
        response = match_template(image_norm, template, pad_input=True)

        # Finde lokale Maxima in der Response
        threshold = 0.3  # Anpassbar
        coords = peak_local_max(
            response,
            min_distance=int(r * 0.8),
            threshold_abs=threshold,
        )

        for (cy, cx) in coords:
            # Response-Wert als Qualitätsmaß
            quality = response[cy, cx]
            all_detections.append({
                'cx': cx,
                'cy': cy,
                'r': r,
                'quality': quality,
                'diam_nm': 2 * r * px_nm,
            })

    print(f"Initiale Detektionen: {len(all_detections)}")

    # --- NMS über alle Skalen ---
    results = []
    circles = []

    for det in all_detections:
        results.append({
            'label': len(results) + 1,
            'x_px': float(det['cx']),
            'y_px': float(det['cy']),
            'refined_radius_px': float(det['r']),
            'refined_diam_px': float(2 * det['r']),
            'refined_diam_nm': float(det['diam_nm']),
            'quality': float(det['quality']),
        })
        circles.append((det['cx'], det['cy'], det['r'], det['quality']))

    # NMS
    results, circles = nms_circles(results, circles, overlap_thresh=0.4)
    print(f"Nach NMS: {len(results)} Partikel")

    # --- Radius-Refinement und Validierung für jeden detektierten Partikel ---
    refined_results = []
    refined_circles = []

    print("Validiere Partikel...")
    for res, (cx, cy, r, q) in zip(results, circles):
        # Verfeinere mit Gradient-Methode
        r_refined, _, _ = refine_radius(image, cx, cy, r)

        # Prüfe ob Refinement sinnvoll
        if r_refined < min_radius_px * 0.5 or r_refined > max_radius_px * 1.5:
            r_refined = r  # Behalte Original

        # --- VALIDIERUNG ---
        validation = validate_particle(image, cx, cy, r_refined)

        # Bei niedrigem Kontrast-Score: versuche Position zu optimieren
        if validation['contrast_score'] < 0.4:
            cx_new, cy_new, improved = refine_particle_position(image, cx, cy, r_refined)
            if improved:
                cx, cy = cx_new, cy_new
                validation = validate_particle(image, cx, cy, r_refined)

        diam_nm = 2 * r_refined * px_nm

        refined_results.append({
            'label': len(refined_results) + 1,
            'x_px': float(cx),
            'y_px': float(cy),
            'refined_radius_px': float(r_refined),
            'refined_diam_px': float(2 * r_refined),
            'refined_diam_nm': float(diam_nm),
            'template_quality': float(q),
            'contrast_score': float(validation['contrast_score']),
            'uniformity_score': float(validation['uniformity_score']),
            'quality_score': float(validation['quality_score']),
            **{f'val_{k}': v for k, v in validation['details'].items()}
        })
        refined_circles.append((cx, cy, r_refined, validation['quality_score']))

    print(f"Final: {len(refined_results)} Partikel")

    # --- CSV speichern (alle Partikel) ---
    df = pd.DataFrame(refined_results)
    output_csv = output_dir / f"segmented_particles_of_{base_name}.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nAlle Ergebnisse gespeichert: {output_csv}")

    # --- INTERAKTIVE Threshold-Visualisierung ---
    # Der Benutzer kann hier die Thresholds für Kontrast und Uniformität einstellen
    # und sieht in Echtzeit welche Partikel akzeptiert werden
    print("\n" + "="*60)
    print("INTERAKTIVE THRESHOLD-AUSWAHL")
    print("="*60)
    print("Verwende die Slider um die Thresholds anzupassen.")
    print("Schließe das Fenster um fortzufahren.")
    print("="*60 + "\n")

    final_thresh_c, final_thresh_u, accepted_particles = interactive_threshold_viewer(
        image_original, refined_results
    )

    # --- Histogramm (nur akzeptierte Partikel) ---
    if accepted_particles:
        fig, ax = plt.subplots(figsize=(8, 5))
        diams = [r['refined_diam_nm'] for r in accepted_particles]
        ax.hist(diams, bins=20, edgecolor='black', color='green', alpha=0.7)
        ax.axvline(np.mean(diams), color='red', linestyle='--',
                  label=f'Mittel: {np.mean(diams):.1f} nm')
        ax.axvline(particle_nm, color='blue', linestyle=':',
                  label=f'Erwartet: {particle_nm:.0f} nm')
        ax.set_xlabel('Durchmesser (nm)')
        ax.set_ylabel('Anzahl')
        ax.set_title(f'Größenverteilung (akzeptierte Partikel, n={len(diams)})\n'
                    f'Mittel: {np.mean(diams):.1f} nm, Std: {np.std(diams):.1f} nm\n'
                    f'Thresholds: Kontrast≥{final_thresh_c:.2f}, Uniformität≥{final_thresh_u:.2f}')
        ax.legend()
        plt.tight_layout()
        histogram_path = output_dir / f"histogram_of_{base_name}.png"
        plt.savefig(histogram_path, dpi=150)
        print(f"Histogramm gespeichert: {histogram_path}")
        plt.show()

    # Akzeptierte Partikel speichern
    df_accepted = pd.DataFrame(accepted_particles)
    accepted_csv = output_dir / f"segmented_particles_of_{base_name}_accepted.csv"
    df_accepted.to_csv(accepted_csv, index=False)
    print(f"Akzeptierte Partikel gespeichert: {accepted_csv}")

    # --- Segmentiertes Bild mit Scale Bar speichern ---
    thresholds = {'contrast': final_thresh_c, 'uniformity': final_thresh_u}

    # Alle Partikel (akzeptiert + abgelehnt sichtbar)
    segmented_path_all = output_dir / f"segmented_of_{base_name}_all.png"
    save_segmented_image(
        image_original, refined_results, px_nm, segmented_path_all,
        title=f"{base_name} - Alle Partikel (n={len(refined_results)})",
        show_accepted_only=False, thresholds=thresholds
    )

    # Nur akzeptierte Partikel
    segmented_path_accepted = output_dir / f"segmented_of_{base_name}_accepted.png"
    save_segmented_image(
        image_original, refined_results, px_nm, segmented_path_accepted,
        title=f"{base_name} - Akzeptierte Partikel (n={len(accepted_particles)})\n"
              f"Kontrast≥{final_thresh_c:.2f}, Uniformität≥{final_thresh_u:.2f}",
        show_accepted_only=True, thresholds=thresholds
    )

    # Durchmesser in µm exportieren (nur akzeptierte)
    if accepted_particles:
        diam_um = np.array([r["refined_diam_nm"] for r in accepted_particles]) / 1000.0
        hist_um_path = output_dir / f"hist_partikel_of_{base_name}_um.txt"
        export_diameter_histogram_um(diam_um, hist_um_path)

    # Rückgabe: alle Ergebnisse, Kreise, akzeptierte Partikel, und gewählte Thresholds
    return {
        'all_particles': refined_results,
        'circles': refined_circles,
        'accepted_particles': accepted_particles,
        'thresholds': {
            'contrast': final_thresh_c,
            'uniformity': final_thresh_u
        },
        'px_nm': px_nm
    }


def nms_circles(results, circles, overlap_thresh=0.5):
    """
    Non-Maximum Suppression für überlappende Kreise.
    Behält den Kreis mit höherer Qualität bei Überlappung.
    """
    if len(circles) == 0:
        return results, circles

    # Sortiere nach Qualität (absteigend)
    indices = sorted(range(len(circles)), key=lambda i: circles[i][3], reverse=True)

    keep = []
    suppressed = set()

    for i in indices:
        if i in suppressed:
            continue

        keep.append(i)
        cx1, cy1, r1, _ = circles[i]

        # Prüfe Überlappung mit allen anderen
        for j in indices:
            if j in suppressed or j == i:
                continue

            cx2, cy2, r2, _ = circles[j]

            # Abstand zwischen Zentren
            dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

            # Überlappung berechnen (IoU-ähnlich für Kreise)
            if dist < (r1 + r2) * overlap_thresh:
                suppressed.add(j)

    # Filtere Ergebnisse
    results_filtered = [results[i] for i in keep]
    circles_filtered = [circles[i] for i in keep]

    return results_filtered, circles_filtered


def sample_circle(image, cx, cy, r, n_theta=64):
    """
    Intensitätsprofil auf einem Kreis mit Radius r um (cx, cy) samplen.
    Bilineare Interpolation via ndi.map_coordinates.
    """
    theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    xs = cx + r * np.cos(theta)
    ys = cy + r * np.sin(theta)
    # Achtung: map_coordinates erwartet Reihenfolge (y, x)
    values = ndi.map_coordinates(image, [ys, xs], order=1, mode="reflect")
    return values
def export_diameter_histogram_um(diam_um, out_path):
    """
    diam_um : 1D-Array / Liste der Partikeldurchmesser in µm
    out_path : Pfad zur Textdatei, z.B. 'hist_partikel.txt'

    - verwendet fixe Bins (in µm, unten definiert)
    - speichert 'bin_min;bin_max;count' mit Komma als Dezimaltrennzeichen
    """

    # --- deine vorgegebenen Bins in µm (mit Punkt als Dezimalpunkt) ---
    bins_um = np.array([
        2.09474E-4, 2.27147E-4, 2.4631E-4, 2.6709E-4, 2.89624E-4,
        3.14058E-4, 3.40554E-4, 3.69285E-4, 4.00440E-4, 4.34224E-4,
        4.70857E-4, 5.10582E-4, 5.53657E-4, 6.00367E-4, 6.51018E-4,
        7.05941E-4, 7.65499E-4, 8.30081E-4, 9.00112E-4, 9.76050E-4,
        0.00106, 0.00115, 0.00124, 0.00135, 0.00146, 0.00159, 0.00172,
        0.00187, 0.00202, 0.00219, 0.00238, 0.00258, 0.00280, 0.00303,
        0.00329, 0.00357, 0.00387, 0.00419, 0.00455, 0.00493, 0.00535,
        0.00580, 0.00629, 0.00682, 0.00739, 0.00802, 0.00869, 0.00943,
        0.01022, 0.01109, 0.01202, 0.01303, 0.01413, 0.01533, 0.01662,
        0.01802, 0.01954, 0.02119, 0.02298, 0.02492, 0.02702, 0.02930,
        0.03177, 0.03445, 0.03736, 0.04051, 0.04393, 0.04763, 0.05165,
        0.05601, 0.06074, 0.06586, 0.07142, 0.07744, 0.08397, 0.09106,
        0.09874, 0.10707, 0.11610, 0.12590, 0.13652, 0.14804, 0.16053,
        0.17407, 0.18876, 0.20468, 0.22195, 0.24068, 0.26098, 0.28300,
        0.30687, 0.33276, 0.36084, 0.39128, 0.42429, 0.46009, 0.49890,
        0.54099, 0.58663, 0.63613, 0.68979, 0.74799, 0.81109, 0.87952,
        0.95372, 1.03418, 1.12143, 1.21604, 1.31864, 1.42989, 1.55052,
        1.68133, 1.82318, 1.97699, 2.14378, 2.32464, 2.52077, 2.73343,
        2.96404, 3.21411, 3.48527, 3.77930, 4.09815, 4.44389, 4.81881,
        5.22535, 5.66619, 6.14423, 6.66259, 7.22469, 7.83420, 8.49514,
        9.21184, 9.98901, 10.83174, 11.74557, 12.73650, 13.81103,
        14.97621, 16.23969, 17.60977, 19.09543
    ])

    diam_um = np.asarray(diam_um, dtype=float)

    # Histogramm
    counts, edges = np.histogram(diam_um, bins=bins_um)

    # Textzeilen vorbereiten (Deutsch-Format: Komma, Semikolon)
    lines = []
    header = "bin_min_um;bin_max_um;count"
    lines.append(header)

    for i, c in enumerate(counts):
        bmin = edges[i]
        bmax = edges[i + 1]

        # in String mit Dezimal-Komma umwandeln
        s_min = f"{bmin:.6g}".replace(".", ",")
        s_max = f"{bmax:.6g}".replace(".", ",")
        line = f"{s_min};{s_max};{int(c)}"
        lines.append(line)

    # Datei schreiben
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Histogramm mit {len(counts)} Bins nach {out_path} geschrieben.")

def refine_radius(image,
                  cx, cy,
                  r_init,
                  factor_inner=0.6,
                  factor_outer=1.4,
                  n_r=40,
                  n_theta=64):
    """
    Lokale Radius-Optimierung um r_init herum.

    - nimmt an: Partikel heller als Hintergrund
    - sucht den Radius mit maximalem negativen dI/dr (stärkster Abfall)
    """
    r_min = r_init * factor_inner
    r_max = r_init * factor_outer
    radii = np.linspace(r_min, r_max, n_r)

    mean_I = []
    for r in radii:
        vals = sample_circle(image, cx, cy, r, n_theta=n_theta)
        mean_I.append(np.mean(vals))
    mean_I = np.array(mean_I)

    # radiale Ableitung dI/dr
    dI = np.diff(mean_I) / np.diff(radii)

    # stärkster negativer Gradient = Kante (hell -> dunkel)
    idx = np.argmin(dI)

    d_mean = (max(mean_I) + min(mean_I)) / 2
    idx_mean = np.where(mean_I < d_mean)[0]

    # Absicherung: falls kein Wert unter dem Mittelwert liegt
    if len(idx_mean) > 0:
        idx = max(idx, idx_mean[0])  # sicherstellen, dass Kante außerhalb des Mittelwerts liegt

    # Absicherung: idx darf nicht der letzte Index sein
    if idx >= len(radii) - 1:
        idx = len(radii) - 2

    r_edge = 0.5 * (radii[idx] + radii[idx + 1])

    return r_edge, radii, mean_I

def circularity(area, perimeter, eps=1e-9):
    return 4.0 * np.pi * area / (perimeter**2 + eps)

def refine_center(image, cx_init, cy_init, r_estimate, search_radius=3, n_theta=64):
    """
    Findet das optimale Zentrum eines Partikels durch Minimierung der Varianz
    im radialen Intensitätsprofil.
    
    Für ein perfekt zentriertes Partikel sollte das radiale Profil bei allen
    Winkeln ähnlich sein. Bei falscher Zentrierung variiert die Intensität
    je nach Winkelrichtung.
    """
    # Grid-Suche für bessere Robustheit
    best_var = np.inf
    best_cx = cx_init
    best_cy = cy_init
    
    # Durchsuche ein kleines Gitter um das initiale Zentrum
    search_steps = 7  # ungerade Zahl für Zentrum
    step_size = search_radius / (search_steps // 2)
    
    for dx in np.linspace(-search_radius, search_radius, search_steps):
        for dy in np.linspace(-search_radius, search_radius, search_steps):
            cx_try = cx_init + dx
            cy_try = cy_init + dy
            
            # Prüfe ob innerhalb des Bildes
            if (cx_try < r_estimate or cx_try > image.shape[1] - r_estimate or
                cy_try < r_estimate or cy_try > image.shape[0] - r_estimate):
                continue
            
            # Sample auf dem geschätzten Radius
            try:
                vals = sample_circle(image, cx_try, cy_try, r_estimate, n_theta=n_theta)
                # Berechne Varianz: je niedriger die Varianz, desto symmetrischer
                var = np.var(vals)
                if var < best_var:
                    best_var = var
                    best_cx = cx_try
                    best_cy = cy_try
            except:
                continue
    
    return best_cx, best_cy

def assess_profile_quality(radii, intensities, r_refined):
    """
    Bewertet die Qualität eines radialen Intensitätsprofils.
    
    Returns:
        dict mit verschiedenen Qualitätsmetriken
    """
    # 1. Gradient-Sharpness am refinierten Radius
    # Finde den Index des refinierten Radius
    idx_refined = np.argmin(np.abs(radii - r_refined))
    if idx_refined > 0 and idx_refined < len(radii) - 1:
        # Gradient am Edge
        dI_dr = (intensities[idx_refined+1] - intensities[idx_refined-1]) / (radii[idx_refined+1] - radii[idx_refined-1])
        gradient_sharpness = abs(dI_dr)
    else:
        gradient_sharpness = 0.0
    
    # 2. Profil-Symmetrie: Varianz der Intensitäten vor dem Peak
    # (sollten relativ konstant sein für ein gutes Partikel)
    idx_before_edge = max(0, idx_refined - 5)
    if idx_before_edge < idx_refined:
        intensities_inner = intensities[idx_before_edge:idx_refined]
        profile_symmetry = 1.0 / (1.0 + np.std(intensities_inner) / (np.mean(intensities_inner) + 1e-9))
    else:
        profile_symmetry = 0.0
    
    # 3. Konsistenz: Wie gut stimmt der refinine Radius mit der initialen Schätzung überein?
    # (wird extern berechnet, da wir den initialen Radius nicht haben)
    
    # 4. Signal-zu-Rausch-Verhältnis am Edge
    # Hoher Kontrast am Edge = gutes Partikel
    if idx_refined < len(intensities) - 3:
        intensity_before = np.mean(intensities[max(0, idx_refined-2):idx_refined])
        intensity_after = np.mean(intensities[idx_refined:min(len(intensities), idx_refined+3)])
        edge_contrast = abs(intensity_before - intensity_after) / (np.max(intensities) + 1e-9)
    else:
        edge_contrast = 0.0
    
    # Kombinierter Qualitäts-Score (0-1, höher ist besser)
    quality_score = 0.4 * min(profile_symmetry, 1.0) + 0.3 * min(gradient_sharpness / np.std(intensities), 1.0) + 0.3 * min(edge_contrast * 5, 1.0)
    
    return {
        'quality_score': quality_score,
        'gradient_sharpness': gradient_sharpness,
        'profile_symmetry': profile_symmetry,
        'edge_contrast': edge_contrast
    }

def check_center_quality(image, cx, cy, r_refined, n_theta=64):
    """
    Prüft die Qualität der Zentrierung durch Analyse der Winkelvarianz.
    
    Ein gut zentriertes Partikel sollte ähnliche Intensitätsprofile
    in allen Richtungen haben.
    """
    # Sample auf verschiedenen Winkeln
    vals = sample_circle(image, cx, cy, r_refined * 0.9, n_theta=n_theta)
    
    # Berechne Varianz: niedrige Varianz = gute Zentrierung
    angular_variance = np.var(vals)
    angular_std = np.std(vals)
    mean_val = np.mean(vals)
    
    # Normalisierte Varianz (Coefficient of Variation)
    cv = angular_std / (mean_val + 1e-9)
    
    # Score: je niedriger CV, desto besser die Zentrierung
    center_quality = 1.0 / (1.0 + cv * 10)
    
    return {
        'center_quality': center_quality,
        'angular_variance': angular_variance,
        'angular_cv': cv
    }
def summarize_sem_metadata(tif_path: str | Path) -> dict:
    """
    Lies ein FEI-SEM-TIF mit Pillow ein und fasse relevante Metadaten zusammen.
    Gibt ein Dictionary mit 'raw' (voll strukturierter FEI-Block)
    und 'summary' (wichtige physikalische Größen) zurück.
    """
    tif_path = Path(tif_path)
    img = Image.open(tif_path)
    tags = img.tag_v2

    # Grundlegende TIFF-Infos
    width, height = img.size
    mode = img.mode

    # FEI-Textblock (tag 34682)
    fei_raw = tags.get(34682, None)
    if fei_raw is None:
        raise RuntimeError("Kein FEI-Metadatenblock (Tag 34682) gefunden.")

    if isinstance(fei_raw, bytes):
        fei_text = fei_raw.decode("utf-8", errors="ignore")
    else:
        fei_text = str(fei_raw)

    sections = parse_fei_text_block(fei_text)

    # Häufig interessante Sektionen
    scan_sec = sections.get("Scan", {})
    ebeam_sec = sections.get("EBeam", {})
    beam_sec = sections.get("Beam", {})
    image_sec = sections.get("Image", {})
    system_sec = sections.get("System", {})

    # FEI arbeitet hier typischerweise in Metern
    # PixelWidth / PixelHeight / HFW / VFW etc. scheinen in m zu sein.
    px_width_m = safe_float(scan_sec, "PixelWidth")
    px_height_m = safe_float(scan_sec, "PixelHeight")

    hfw_m = safe_float(scan_sec, "HorFieldsize") or safe_float(ebeam_sec, "HFW")
    vfw_m = safe_float(scan_sec, "VerFieldsize") or safe_float(ebeam_sec, "VFW")

    dwell_s = safe_float(scan_sec, "Dwelltime") or safe_float(scan_sec, "Dwell")
    hv_V = safe_float(ebeam_sec, "HV") or safe_float(beam_sec, "HV")
    wd_m = safe_float(ebeam_sec, "WD")

    # Einheiten-Umrechnungen
    px_width_nm = px_width_m * 1e9 if px_width_m is not None else None
    px_height_nm = px_height_m * 1e9 if px_height_m is not None else None

    hfw_um = hfw_m * 1e6 if hfw_m is not None else None
    vfw_um = vfw_m * 1e6 if vfw_m is not None else None

    hv_kV = hv_V / 1000.0 if hv_V is not None else None
    wd_mm = wd_m * 1e3 if wd_m is not None else None

    summary = {
        "file": str(tif_path),
        "image_size_px": (width, height),
        "image_mode": mode,

        # Pixelgröße
        "pixel_width_m": px_width_m,
        "pixel_height_m": px_height_m,
        "pixel_width_nm": px_width_nm,
        "pixel_height_nm": px_height_nm,

        # Field of View
        "HFW_m": hfw_m,
        "VFW_m": vfw_m,
        "HFW_um": hfw_um,
        "VFW_um": vfw_um,

        # Elektronenstrahl-Parameter
        "HV_V": hv_V,
        "HV_kV": hv_kV,
        "WD_m": wd_m,
        "WD_mm": wd_mm,

        # Scanparameter
        "dwell_time_s": dwell_s,
        "frame_time_s": safe_float(scan_sec, "FrameTime"),

        # Sonstige interessante Angaben
        "magnification_correction": beam_sec.get("MagnificationCorrection", None),
        "HFW_from_EBeam_m": safe_float(ebeam_sec, "HFW"),
        "VFW_from_EBeam_m": safe_float(ebeam_sec, "VFW"),
        "scan_rotation_rad": safe_float(ebeam_sec, "ScanRotation"),
        "stage_X_m": safe_float(ebeam_sec, "StageX"),
        "stage_Y_m": safe_float(ebeam_sec, "StageY"),
        "stage_Z_m": safe_float(ebeam_sec, "StageZ"),
        "software_version": system_sec.get("Software", None),
        "microscope_type": system_sec.get("Type", None),
    }

    return {
        "raw_sections": sections,   # vollständig geparst, falls du später noch was brauchst
        "summary": summary,         # komprimierte, physikalisch relevante Infos
    }

def parse_fei_text_block(text: str) -> dict:
    """
    Parsen des FEI-Textblocks aus Tag 34682.
    Struktur:
        [Section]
        key=value
        ...

    Rückgabe:
        { "Section": { "key": "value", ... }, ... }
    """
    sections = {}
    current_section = None

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Neue Sektion
        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1].strip()
            sections[current_section] = {}
            continue

        # key=value Zeilen
        if "=" in line and current_section is not None:
            key, value = line.split("=", 1)
            sections[current_section][key.strip()] = value.strip()

    return sections


def safe_float(d: dict, key: str):
    """Hilfsfunktion: float-Wert aus dict, oder None."""
    val = d.get(key, None)
    if val is None or val == "":
        return None
    try:
        return float(val)
    except ValueError:
        return None
def segment_particles_watershed(tif_path, particle_nm, size_factor_min=0.8, size_factor_max=1.2):
    """
    Partikelsegmentierung mit Watershed-Methode.

    Args:
        tif_path: Pfad zum TIFF-Bild
        particle_nm: Erwartete Partikelgröße in nm (aus Dateiname/Pfad extrahiert)
        size_factor_min: Faktor für minimale Partikelgröße (default: 0.6)
        size_factor_max: Faktor für maximale Partikelgröße (default: 1.5)
    """
    tif_path = Path(tif_path)

    # --- Ausgabe-Verzeichnis und Basis-Dateiname ---
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = tif_path.stem  # Dateiname ohne Erweiterung

    # --- Erwartete Partikelgröße aus übergebenem Parameter ---
    expected_diam_min = particle_nm * size_factor_min
    expected_diam_max = particle_nm * size_factor_max

    # --- Pixelgröße aus FEI-Metadaten (falls vorhanden) -------------
    meta = summarize_sem_metadata(tif_path)
    px_nm = 0.5*(meta["summary"]["pixel_width_nm"] +
                 meta["summary"]["pixel_height_nm"])
    
    if px_nm is None or px_nm <= 0:
        raise ValueError("Pixelgröße konnte nicht aus Metadaten extrahiert werden. "
                        "Bitte prüfe die TIFF-Metadaten.")
    
    print(f"\n{'='*60}")
    print(f"WATERSHED SEGMENTIERUNG")
    print(f"{'='*60}")
    print(f"Datei: {tif_path.name}")
    print(f"Pixelgröße: {px_nm:.4f} nm/pixel")
    print(f"Erwartete Partikelgröße: {expected_diam_min:.0f} - {expected_diam_max:.0f} nm (Nominal: {particle_nm} nm)")
    print(f"{'='*60}\n")

    # --- Automatische Berechnung der Pixel-basierten Parameter aus nm-Werten ---
    # Minimale und maximale Partikelradien in Pixel
    min_radius_nm = expected_diam_min / 2.0
    max_radius_nm = expected_diam_max / 2.0
    min_radius_px = min_radius_nm / px_nm
    max_radius_px = max_radius_nm / px_nm

    # Flächenberechnung in Pixel
    MIN_AREA_PX = int(MIN_AREA_FACTOR * np.pi * min_radius_px**2)
    MAX_AREA_PX = int(MAX_AREA_FACTOR * np.pi * max_radius_px**2)

    # Gaussian Sigma: sollte etwa 5-10% des minimalen Partikeldurchmessers entsprechen
    # Für bessere Anpassung: ca. 2-3% des Durchmessers oder min. 2-3 Pixel
    if GAUSSIAN_SIGMA_OVERRIDE is not None:
        GAUSSIAN_SIGMA = GAUSSIAN_SIGMA_OVERRIDE
    else:
        # Automatisch basierend auf Partikelgröße: ca. 2-3% des min. Durchmessers
        gaussian_sigma_nm = expected_diam_min * 0.025  # 2.5% des Durchmessers
        GAUSSIAN_SIGMA = max(gaussian_sigma_nm / px_nm, 1.5)  # Mindestens 1.5 Pixel
        GAUSSIAN_SIGMA = min(GAUSSIAN_SIGMA, 10.0)  # Maximal 10 Pixel
    
    # MAX_CENTER_OFFSET in Pixel aus nm-Wert berechnen
    MAX_CENTER_OFFSET_PX = MAX_CENTER_OFFSET_NM / px_nm
    
    print(f"Berechnete Parameter (in Pixel):")
    print(f"  Min. Radius: {min_radius_px:.2f} px")
    print(f"  Max. Radius: {max_radius_px:.2f} px")
    print(f"  Min. Fläche: {MIN_AREA_PX:.0f} px²")
    print(f"  Max. Fläche: {MAX_AREA_PX:.0f} px²")
    print(f"  Gaussian Sigma: {GAUSSIAN_SIGMA:.2f} px")
    print(f"  Max. Zentrums-Offset: {MAX_CENTER_OFFSET_PX:.2f} px")
    print()

    # --- Bild laden & in Graustufen -------------------------------
    pil_img = Image.open(tif_path)
    if np.max(pil_img) < 255:
        img_gray = pil_img.convert("L")
    else:
        img_gray = pil_img
    image = np.array(img_gray, dtype=float)

    # --- 1) Denoising ---------------------------------------------
    smoothed = gaussian(image, sigma=GAUSSIAN_SIGMA)

    # --- 2) Threshold + Maske --------------------------------------
    thr = threshold_otsu(smoothed)
    mask = smoothed > thr

    # kleine Löcher schließen und Noise entfernen
    mask = closing(mask, footprint=disk(1))
    mask = remove_small_objects(mask, min_size=MIN_AREA_PX)

    # --- 3) Distance-Transform -------------------------------------
    distance = ndi.distance_transform_edt(mask)

    # --- 4) Peaks im Distance-Map (Seeds) --------------------------
    # min_distance sollte etwa 50% des minimalen Partikelradius sein
    # Verhindert, dass zwei Partikel zu nah beieinander als einer erkannt werden
    min_dist = max(int(min_radius_px * 0.5), 2)

    coords = peak_local_max(
        distance,
        min_distance=min_dist,
        labels=mask
    )

    markers = np.zeros_like(image, dtype=int)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i

    # --- 5) Watershed-Segmentierung -------------------------------
    labels_ws = watershed(-distance, markers, mask=mask)

    # --- 6) Analyse der Regionen ----------------------------------
    props = regionprops(labels_ws, intensity_image=image)
    s = 0
    results = []
    # Liste zum Sammeln von Profilen für kleine Partikel (< 120 nm)
    small_particle_profiles = []
    
    # Statistiken für Filterung
    filter_stats = {
        'total_detected': 0,
        'filtered_area': 0,
        'filtered_circularity': 0,
        'filtered_center_offset': 0,
        'filtered_radius_change': 0,
        'filtered_quality_score': 0,
        'filtered_center_quality': 0,
        'accepted': 0
    }
    
    for r in props:
        filter_stats['total_detected'] += 1
        area = r.area
        if not (MIN_AREA_PX <= area <= MAX_AREA_PX):
            filter_stats['filtered_area'] += 1
            if DEBUG_FILTERING:
                print(f"  Partikel {r.label}: Herausgefiltert wegen Bereich (Area: {area:.1f}, erwartet: {MIN_AREA_PX:.1f}-{MAX_AREA_PX:.1f})")
            continue

        circ = circularity(area, r.perimeter)
        if circ < MIN_CIRCULARITY:
            filter_stats['filtered_circularity'] += 1
            if DEBUG_FILTERING:
                print(f"  Partikel {r.label}: Herausgefiltert wegen Zirkularität ({circ:.3f} < {MIN_CIRCULARITY:.3f})")
            continue

        eq_diam_px = r.equivalent_diameter
        cy_init, cx_init = r.centroid
        
        # --- Zentrum-Refinement: Finde das optimale Zentrum vor Radius-Refinement ---
        r0_estimate = eq_diam_px / 2.0
        # Suchradius: entweder MAX_CENTER_OFFSET_PX oder 30% des geschätzten Radius, je nachdem was kleiner ist
        search_radius = min(MAX_CENTER_OFFSET_PX, r0_estimate * 0.3)
        cx_refined, cy_refined = refine_center(
            image, cx_init, cy_init, r0_estimate, 
            search_radius=search_radius, 
            n_theta=64
        )
        
        # Prüfe, ob Zentrum-Offset zu groß ist (möglicherweise schlechtes Partikel)
        center_offset = np.sqrt((cx_refined - cx_init)**2 + (cy_refined - cy_init)**2)
        if center_offset > MAX_CENTER_OFFSET_PX:
            filter_stats['filtered_center_offset'] += 1
            if DEBUG_FILTERING:
                print(f"  Partikel {r.label}: Herausgefiltert wegen zu großem Zentrums-Offset ({center_offset:.2f} > {MAX_CENTER_OFFSET_PX:.2f} px)")
            continue  # Überspringe Partikel mit zu großem Zentrums-Offset
        
        # --- Radius-Refinement mit verbessertem Zentrum ---
        r_refined, rad_arr, prof = refine_radius(
            image, cx_refined, cy_refined, r0_estimate
        )
        radii, I = rad_arr, prof  # aus refine_radius zurückgegeben
        
        # Prüfe, ob Radius-Änderung zu groß ist (möglicherweise schlechtes Fitting)
        radius_change_factor = abs(r_refined - r0_estimate) / (r0_estimate + 1e-9)
        if radius_change_factor > MAX_RADIUS_CHANGE_FACTOR:
            filter_stats['filtered_radius_change'] += 1
            if DEBUG_FILTERING:
                print(f"  Partikel {r.label}: Herausgefiltert wegen zu großer Radius-Änderung ({radius_change_factor:.2%} > {MAX_RADIUS_CHANGE_FACTOR:.2%})")
            continue  # Überspringe Partikel mit zu großer Radius-Änderung
        
        # --- Qualitätsbewertung ---
        profile_quality = assess_profile_quality(radii, I, r_refined)
        center_quality = check_center_quality(image, cx_refined, cy_refined, r_refined, n_theta=64)
        
        # Filtere Partikel mit schlechter Qualität
        if profile_quality['quality_score'] < MIN_QUALITY_SCORE:
            filter_stats['filtered_quality_score'] += 1
            if DEBUG_FILTERING:
                print(f"  Partikel {r.label}: Herausgefiltert wegen niedrigem Quality Score ({profile_quality['quality_score']:.3f} < {MIN_QUALITY_SCORE:.3f})")
            continue  # Überspringe Partikel mit schlechter Profilqualität
            
        if center_quality['center_quality'] < MIN_CENTER_QUALITY:
            filter_stats['filtered_center_quality'] += 1
            if DEBUG_FILTERING:
                print(f"  Partikel {r.label}: Herausgefiltert wegen niedriger Zentrierungsqualität ({center_quality['center_quality']:.3f} < {MIN_CENTER_QUALITY:.3f})")
            continue  # Überspringe Partikel mit schlechter Zentrierung
        
        filter_stats['accepted'] += 1
        
        # Verwende die verfeinerten Koordinaten
        cx, cy = cx_refined, cy_refined
        
        # Umrechnung von Pixel zu nm
        radii_nm = radii * px_nm
        r_refined_nm = r_refined * px_nm
        eq_refined_nm = 2.0 * r_refined_nm  # Durchmesser in nm
        
        # Sammle Profile für kleine Partikel (< 120 nm) zum Overlay
        if eq_refined_nm < 300:
            small_particle_profiles.append({
                'radii_nm': radii_nm,
                'I': I,
                'r_refined_nm': r_refined_nm,
                'cx': cx,
                'cy': cy,
                'diameter_nm': eq_refined_nm,
                'label': r.label
            })
        else:
            # Detaillierte Plots für größere Partikel (>= 120 nm)
            if False :  # Nur für das erste große Partikel
                # Erstelle eine Figur mit zwei Subplots: Profil + Bildausschnitt
                fig = plt.figure(figsize=(14, 6))
                
                # Linker Subplot: Radiales Intensitätsprofil (in nm)
                ax1 = plt.subplot(1, 2, 1)
                ax1.plot(radii_nm, I, "-o", markersize=4, label="Intensitätsprofil")
                ax1.scatter([r_refined_nm], [np.interp(r_refined, radii, I)], 
                           color="red", s=100, zorder=5, label=f"Refinierter Radius: {r_refined_nm:.2f} nm")
                ax1.axvline(r_refined_nm, color="red", linestyle="--", alpha=0.5, linewidth=1)
                ax1.set_xlabel("Radius [nm]")
                ax1.set_ylabel("⟨I⟩ auf Kreis")
                ax1.set_title(f"Radiales Intensitätsprofil\nPosition: ({cx:.1f}, {cy:.1f}) px")
                ax1.grid(True, alpha=0.3)
                ax1.legend()
                
                # Rechter Subplot: Bildausschnitt mit markierter Position
                ax2 = plt.subplot(1, 2, 2)
                
                # Definiere Ausschnitt um die Partikelposition (3x Radius in jede Richtung)
                crop_size = int(r_refined * 3)
                y_min = max(0, int(cy - crop_size))
                y_max = min(image.shape[0], int(cy + crop_size))
                x_min = max(0, int(cx - crop_size))
                x_max = min(image.shape[1], int(cx + crop_size))
                
                image_crop = image[y_min:y_max, x_min:x_max]
                ax2.imshow(image_crop, cmap="gray", interpolation="nearest")
                
                # Position relativ zum Ausschnitt
                cx_rel = cx - x_min
                cy_rel = cy - y_min
                
                # Zeichne den refinierten Kreis
                circle = plt.Circle((cx_rel, cy_rel), r_refined, fill=False, 
                                  color="red", linewidth=2, label=f"Radius: {r_refined_nm:.2f} nm")
                ax2.add_patch(circle)
                
                # Markiere den Mittelpunkt
                ax2.plot(cx_rel, cy_rel, "r+", markersize=15, markeredgewidth=2, 
                        label=f"Zentrum: ({cx:.1f}, {cy:.1f}) px")
                
                ax2.set_title(f"Bildausschnitt mit segmentiertem Kreis\nAbsolute Position: ({cx:.2f}, {cy:.2f}) px")
                ax2.legend(loc="upper right", fontsize=9)
                ax2.axis("off")
                
                # Zusätzliche Informationen als Text
                info_text = f"Radius: {r_refined_nm:.3f} nm ({r_refined:.2f} px)\n"
                info_text += f"Durchmesser: {eq_refined_nm:.3f} nm ({2*r_refined:.2f} px)\n"
                info_text += f"Position: ({cx:.2f}, {cy:.2f}) px"
                fig.text(0.5, 0.02, info_text, ha="center", va="bottom", 
                        fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
                
                plt.tight_layout()
                plt.show()
                
                s += 1
        
        eq_refined_px = 2.0 * r_refined
        eq_refined_nm = eq_refined_px * px_nm  # px_nm kommt aus deinen Metadaten

        # --- NEUE VALIDIERUNG ---
        particle_validation = validate_particle(image, cx, cy, r_refined)

        # Bei niedrigem Kontrast-Score: versuche Position zu optimieren
        if particle_validation['contrast_score'] < 0.4:
            cx_new, cy_new, improved = refine_particle_position(image, cx, cy, r_refined)
            if improved:
                cx, cy = cx_new, cy_new
                particle_validation = validate_particle(image, cx, cy, r_refined)

        results.append(
            dict(
                label=int(r.label),
                x_px=float(cx),
                y_px=float(cy),
                x_init_px=float(cx_init),  # Initiales Zentrum
                y_init_px=float(cy_init),
                center_offset_px=float(center_offset),
                area_px=float(area),
                circularity=float(circ),
                eq_diam_px=float(eq_diam_px),
                eq_diam_nm=float(eq_diam_px * px_nm),
                refined_diam_px=float(eq_refined_px),
                refined_diam_nm=float(eq_refined_nm),
                refined_radius_px=float(r_refined),
                radius_change_factor=float(radius_change_factor),
                contrast_score=float(particle_validation['contrast_score']),
                uniformity_score=float(particle_validation['uniformity_score']),
                quality_score=float(particle_validation['quality_score']),
                gradient_sharpness=float(profile_quality['gradient_sharpness']),
                profile_symmetry=float(profile_quality['profile_symmetry']),
                edge_contrast=float(profile_quality['edge_contrast']),
                center_quality=float(center_quality['center_quality']),
                angular_variance=float(center_quality['angular_variance']),
                angular_cv=float(center_quality['angular_cv']),
                mean_intensity=float(r.mean_intensity),
                max_intensity=float(r.max_intensity),
                **{f'val_{k}': v for k, v in particle_validation['details'].items()}
            )
        )
    
    # --- Filterstatistik ausgeben ---
    print("\n" + "="*60)
    print("FILTERSTATISTIKEN:")
    print("="*60)
    print(f"Gesamt detektierte Partikel: {filter_stats['total_detected']}")
    print(f"  ✓ Akzeptiert: {filter_stats['accepted']}")
    print(f"  ✗ Herausgefiltert:")
    print(f"     - Bereich (Area): {filter_stats['filtered_area']}")
    print(f"     - Zirkularität: {filter_stats['filtered_circularity']}")
    print(f"     - Zentrums-Offset: {filter_stats['filtered_center_offset']}")
    print(f"     - Radius-Änderung: {filter_stats['filtered_radius_change']}")
    print(f"     - Quality Score: {filter_stats['filtered_quality_score']}")
    print(f"     - Zentrierungsqualität: {filter_stats['filtered_center_quality']}")
    print("="*60 + "\n")
    
    # --- Overlay-Plot für kleine Partikel (< 120 nm) ---
    if small_particle_profiles:
        fig_overlay = plt.figure(figsize=(14, 6))
        
        # Linker Subplot: Overlay aller radialen Intensitätsprofile
        ax1 = plt.subplot(1, 2, 1)
        
        # Farbschema für mehrere Partikel
        colors = plt.cm.tab10(np.linspace(0, 1, min(len(small_particle_profiles), 10)))
        
        for idx, prof_data in enumerate(small_particle_profiles):
            color = colors[idx % len(colors)]
            label = f"Partikel {prof_data['label']}: {prof_data['diameter_nm']:.1f} nm"
            ax1.plot(prof_data['radii_nm'], prof_data['I'], "-o", markersize=3, 
                    alpha=0.7, color=color, label=label, linewidth=1.5)
            # Markiere den refinierten Radius für jedes Partikel
            # Finde den Index des refinierten Radius oder interpoliere
            r_refined_px = prof_data['r_refined_nm'] / px_nm
            radii_px = prof_data['radii_nm'] / px_nm
            I_at_refined = np.interp(r_refined_px, radii_px, prof_data['I'])
            ax1.scatter([prof_data['r_refined_nm']], [I_at_refined],
                       color=color, s=80, zorder=5, marker='x', linewidths=2)
            ax1.axvline(prof_data['r_refined_nm'], color=color, linestyle="--", 
                       alpha=0.4, linewidth=1)
        
        ax1.set_xlabel("Radius [nm]", fontsize=11)
        ax1.set_ylabel("⟨I⟩ auf Kreis", fontsize=11)
        ax1.set_title(f"Overlay: Radiale Intensitätsprofile\n{len(small_particle_profiles)} Partikel < 120 nm", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=8, ncol=1)
        
        # Rechter Subplot: Bild mit allen markierten kleinen Partikeln
        ax2 = plt.subplot(1, 2, 2)
        ax2.imshow(image, cmap="gray", interpolation="nearest")
        
        # Zeichne alle kleinen Partikel mit ihren refinierten Kreisen
        for idx, prof_data in enumerate(small_particle_profiles):
            color = colors[idx % len(colors)]
            r_refined_px = prof_data['r_refined_nm'] / px_nm
            circle = plt.Circle((prof_data['cx'], prof_data['cy']), r_refined_px, 
                              fill=False, color=color, linewidth=1.5, alpha=0.8)
            ax2.add_patch(circle)
            # Markiere den Mittelpunkt
            ax2.plot(prof_data['cx'], prof_data['cy'], "+", color=color, 
                    markersize=10, markeredgewidth=2, alpha=0.9)
        
        ax2.set_title(f"Bild mit allen kleinen Partikeln markiert\n({len(small_particle_profiles)} Partikel < 120 nm)", fontsize=12)
        ax2.axis("off")
        
        # Zusammenfassungstext
        avg_diameter = np.mean([p['diameter_nm'] for p in small_particle_profiles])
        std_diameter = np.std([p['diameter_nm'] for p in small_particle_profiles])
        summary_text = f"Kleine Partikel (< 120 nm): {len(small_particle_profiles)}\n"
        summary_text += f"Durchschnittlicher Durchmesser: {avg_diameter:.2f} ± {std_diameter:.2f} nm"
        fig_overlay.text(0.5, 0.02, summary_text, ha="center", va="bottom", 
                        fontsize=10, bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7))
        
        plt.tight_layout()
        plt.show()
        
        print(f"\n{len(small_particle_profiles)} kleine Partikel (< 120 nm) gefunden und im Overlay-Plot dargestellt.")
    
    # --- 7) Filtern der Ergebnisse nach physikalisch sinnvollen Größen ---
    # Filteriere direkt die Liste `results` statt das DataFrame `df`
    filtered_results = [
        r for r in results
        if (r["circularity"] >= MIN_CIRCULARITY)
        and (r["circularity"] <= MAX_CIRCULARITY)
        and (r["max_intensity"] >= MIN_INTENSITYMAX)]

    if not results:
        print("Keine Partikel gefunden – Threshold/Parameter anpassen.")
        return

    # --- CSV speichern (alle Partikel) ---
    df = pd.DataFrame(results)
    output_csv = output_dir / f"segmented_particles_of_{base_name}.csv"
    df.to_csv(output_csv, index=False)
    print(f"{len(df)} Partikel gefunden. Ergebnisse in: {output_csv}")

    # Optional: gefilterte Partikel speichern
    filtered_df = pd.DataFrame(filtered_results)
    filtered_csv = output_dir / f"segmented_particles_of_{base_name}_filtered.csv"
    filtered_df.to_csv(filtered_csv, index=False)

    print(f"{len(filtered_results)} Partikel nach Filterung. Gefilterte Ergebnisse in: {filtered_csv}")

    # Für weitere Verarbeitung/Plotting die gefilterte Liste verwenden
    results = filtered_results

    # --- 8) Overlay ------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Binärmaske")
    axes[1].axis("off")

    axes[2].imshow(image, cmap="gray")
    for r in results:
        x = r["x_px"]
        y = r["y_px"]
        rad = r["eq_diam_px"] / 2.0
        circ_patch = plt.Circle((x, y), rad, fill=False, linewidth=1, color='red')
        axes[2].add_patch(circ_patch)
    axes[2].set_title("Segmentierte Partikel (Kreise = eq. Durchmesser)")
    axes[2].axis("off")
    plt.tight_layout()
    plt.show()

    # --- INTERAKTIVE Threshold-Visualisierung ---
    print("\n" + "="*60)
    print("INTERAKTIVE THRESHOLD-AUSWAHL")
    print("="*60)
    print("Verwende die Slider um die Thresholds anzupassen.")
    print("Schließe das Fenster um fortzufahren.")
    print("="*60 + "\n")

    final_thresh_c, final_thresh_u, accepted_particles = interactive_threshold_viewer(
        image, results
    )

    # --- Histogramm nur für akzeptierte Partikel ---
    if accepted_particles:
        plt.figure(figsize=(8, 5))
        diams = [r["refined_diam_nm"] for r in accepted_particles]
        plt.hist(diams, bins=30, edgecolor='black', color='green', alpha=0.7)
        plt.axvline(np.mean(diams), color='red', linestyle='--',
                   label=f'Mittel: {np.mean(diams):.1f} nm')
        plt.xlabel("Durchmesser [nm]")
        plt.ylabel("Anzahl")
        plt.title(f"Größenverteilung (akzeptierte Partikel, n={len(diams)})\n"
                 f"Mittel: {np.mean(diams):.1f} nm, Std: {np.std(diams):.1f} nm\n"
                 f"Thresholds: Kontrast≥{final_thresh_c:.2f}, Uniformität≥{final_thresh_u:.2f}")
        plt.legend()
        plt.tight_layout()
        histogram_path = output_dir / f"histogram_of_{base_name}.png"
        plt.savefig(histogram_path, dpi=150)
        print(f"Histogramm gespeichert: {histogram_path}")
        plt.show()

    # Durchmesser in µm ableiten (nur akzeptierte)
    if accepted_particles:
        diam_um = np.array([r["refined_diam_nm"] for r in accepted_particles]) / 1000.0
        hist_um_path = output_dir / f"hist_partikel_of_{base_name}_um.txt"
        export_diameter_histogram_um(diam_um, hist_um_path)

    # Akzeptierte Partikel speichern
    df_accepted = pd.DataFrame(accepted_particles)
    accepted_csv = output_dir / f"segmented_particles_of_{base_name}_accepted.csv"
    df_accepted.to_csv(accepted_csv, index=False)
    print(f"Akzeptierte Partikel gespeichert: {accepted_csv}")

    # --- Segmentiertes Bild mit Scale Bar speichern ---
    thresholds = {'contrast': final_thresh_c, 'uniformity': final_thresh_u}

    # Alle Partikel (akzeptiert + abgelehnt sichtbar)
    segmented_path_all = output_dir / f"segmented_of_{base_name}_all.png"
    save_segmented_image(
        image, results, px_nm, segmented_path_all,
        title=f"{base_name} - Alle Partikel (n={len(results)})",
        show_accepted_only=False, thresholds=thresholds
    )

    # Nur akzeptierte Partikel
    segmented_path_accepted = output_dir / f"segmented_of_{base_name}_accepted.png"
    save_segmented_image(
        image, results, px_nm, segmented_path_accepted,
        title=f"{base_name} - Akzeptierte Partikel (n={len(accepted_particles)})\n"
              f"Kontrast≥{final_thresh_c:.2f}, Uniformität≥{final_thresh_u:.2f}",
        show_accepted_only=True, thresholds=thresholds
    )

    # Rückgabe: alle Ergebnisse, akzeptierte Partikel, und gewählte Thresholds
    return {
        'all_particles': results,
        'accepted_particles': accepted_particles,
        'thresholds': {
            'contrast': final_thresh_c,
            'uniformity': final_thresh_u
        },
        'px_nm': px_nm
    }
def get_expected_output_paths(tif_path):
    tif_path = Path(tif_path)
    output_dir = OUTPUT_DIR
    base_name = tif_path.stem
    return [
        output_dir / f"segmented_particles_of_{base_name}.csv",
        output_dir / f"segmented_particles_of_{base_name}_accepted.csv",
        output_dir / f"segmented_of_{base_name}_all.png",
        output_dir / f"segmented_of_{base_name}_accepted.png",
    ]


if __name__ == "__main__":
    # ============================================================================
    # EINGABE: Ordner oder einzelne Datei
    # ============================================================================
    input_path = r"D:\SEM"

    # Wähle Methode:
    # - 'template': Template Matching (empfohlen für bekannte Partikelgrößen)
    # - 'watershed': Watershed-Methode (gut für gut separierte Partikel)
    METHOD = 'template'  # 'template' oder 'watershed'

    # ============================================================================
    # BATCH-VERARBEITUNG
    # ============================================================================
    input_path = Path(input_path)

    # Sammle alle TIF-Dateien
    if input_path.is_dir():
        # Ordner: Finde alle .tif Dateien (rekursiv)
        tif_files = list(input_path.rglob("*.tif")) + list(input_path.rglob("*.TIF"))
        print(f"\n{'='*70}")
        print(f"BATCH-VERARBEITUNG: {len(tif_files)} TIF-Dateien gefunden")
        print(f"Ordner: {input_path}")
        print(f"{'='*70}\n")
    elif input_path.is_file() and input_path.suffix.lower() == '.tif':
        # Einzelne Datei
        tif_files = [input_path]
        print(f"\nEinzeldatei: {input_path}\n")
    else:
        raise ValueError(f"Ungültiger Pfad: {input_path}")

    # Ergebnisse sammeln
    results_summary = []
    failed_files = []
    skipped_files = []

    for i, tif_path in enumerate(tif_files, 1):
        print(f"\n{'#'*70}")
        print(f"# DATEI {i}/{len(tif_files)}: {tif_path.name}")
        print(f"# Pfad: {tif_path}")
        print(f"{'#'*70}")

        expected_outputs = get_expected_output_paths(tif_path)
        if all(p.exists() for p in expected_outputs):
            print("Ausgabe existiert bereits (CSV/PNG). Ueberspringe Datei.")
            skipped_files.append(tif_path.name)
            results_summary.append({
                'file': tif_path.name,
                'status': 'SKIPPED',
                'reason': 'outputs_exist'
            })
            continue

        try:
            # Partikelgröße aus Dateiname/Pfad extrahieren
            particle_nm = extract_particle_size_from_path(tif_path)
            print(f"Erkannte Partikelgröße: {particle_nm} nm")

            # Segmentierung durchführen (particle_nm wird übergeben)
            if METHOD == 'template':
                result = segment_particles_template_matching(tif_path, particle_nm=particle_nm, n_scales=24)
            else:
                result = segment_particles_watershed(tif_path, particle_nm=particle_nm)

            # Ergebnis speichern
            if result:
                n_all = len(result['all_particles'])
                n_accepted = len(result['accepted_particles'])
                results_summary.append({
                    'file': tif_path.name,
                    'particle_size_nm': particle_nm,
                    'total_detected': n_all,
                    'accepted': n_accepted,
                    'acceptance_rate': n_accepted / n_all * 100 if n_all > 0 else 0,
                    'contrast_threshold': result['thresholds']['contrast'],
                    'uniformity_threshold': result['thresholds']['uniformity'],
                    'status': 'OK'
                })

        except Exception as e:
            print(f"\n!!! FEHLER bei {tif_path.name}: {e}")
            failed_files.append({'file': tif_path.name, 'error': str(e)})
            results_summary.append({
                'file': tif_path.name,
                'status': 'FEHLER',
                'error': str(e)
            })

    # ============================================================================
    # ZUSAMMENFASSUNG
    # ============================================================================
    print(f"\n\n{'='*70}")
    print("BATCH-VERARBEITUNG ABGESCHLOSSEN")
    print(f"{'='*70}")
    successful = len(tif_files) - len(failed_files) - len(skipped_files)
    print(f"Erfolgreich: {successful}/{len(tif_files)} Dateien")

    if failed_files:
        print(f"\nFehlgeschlagen ({len(failed_files)}):")
        for f in failed_files:
            print(f"  - {f['file']}: {f['error']}")

    if skipped_files:
        print(f"\nUebersprungen ({len(skipped_files)}):")
        for f in skipped_files:
            print(f"  - {f}")

    # Zusammenfassung als CSV speichern
    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        summary_path = OUTPUT_DIR / "batch_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nZusammenfassung gespeichert: {summary_path}")

    print(f"{'='*70}\n")



