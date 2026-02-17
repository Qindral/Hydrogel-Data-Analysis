"""Qt-basierte Benutzeroberfläche für die SEM-Partikelauswertung."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy import QtCore, QtGui, QtWidgets
from skimage import util

from sem_particle_analysis import ParticleStatistics, create_overlay, perform_analysis, load_image_with_dpi


def _ensure_float_image(image: np.ndarray) -> np.ndarray:
    if image.dtype == bool:
        return image.astype(np.float32)
    if image.dtype.kind in {"f"}:
        return image
    return util.img_as_float(image)


def _to_qimage(image: np.ndarray) -> QtGui.QImage:
    if image.ndim == 2:
        array = util.img_as_ubyte(_ensure_float_image(image))
        height, width = array.shape
        bytes_per_line = array.strides[0]
        qimage = QtGui.QImage(array.data, width, height, bytes_per_line, QtGui.QImage.Format_Grayscale8)
        qimage.ndarray = array  # type: ignore[attr-defined]
        return qimage

    if image.ndim == 3 and image.shape[2] == 3:
        array = util.img_as_ubyte(_ensure_float_image(image))
        height, width, _channels = array.shape
        bytes_per_line = array.strides[0]
        qimage = QtGui.QImage(array.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
        qimage.ndarray = array  # type: ignore[attr-defined]
        return qimage

    raise ValueError("Nur 2-D- oder 3-Kanal-RGB-Bilder werden unterstützt.")


class ImageDisplay(QtWidgets.QLabel):
    """Einfache Bildanzeige innerhalb des UI."""

    def __init__(self, placeholder: str, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent=parent)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setText(placeholder)
        self.setMinimumSize(320, 240)
        self._buffer: Optional[np.ndarray] = None
        self._original_pixmap: Optional[QtGui.QPixmap] = None

    def set_array(self, array: Optional[np.ndarray]) -> None:
        if array is None:
            self._buffer = None
            self._original_pixmap = None
            self.setPixmap(QtGui.QPixmap())
            self.setText("Kein Bild verfügbar")
            return

        qimage = _to_qimage(array)
        self._buffer = getattr(qimage, "ndarray", None)
        self._original_pixmap = QtGui.QPixmap.fromImage(qimage)
        self._update_pixmap()
        self.setText("")

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: D401 - Qt API
        super().resizeEvent(event)
        self._update_pixmap()

    def _update_pixmap(self) -> None:
        if self._original_pixmap is None or self._original_pixmap.isNull():
            return
        scaled = self._original_pixmap.scaled(
            self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
        self.setPixmap(scaled)


class HistogramCanvas(FigureCanvasQTAgg):
    """Matplotlib-Canvas zur Darstellung der Durchmesserverteilung."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        self._figure = Figure(figsize=(4, 3))
        super().__init__(self._figure)
        self.setParent(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.update_histogram(None, None, None)

    def update_histogram(
        self,
        diameters_um: Optional[np.ndarray],
        histogram: Optional[tuple[np.ndarray, np.ndarray]],
        stats: Optional[ParticleStatistics],
    ) -> None:
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        if diameters_um is None or histogram is None or len(diameters_um) == 0:
            ax.text(0.5, 0.5, "Keine Daten", ha="center", va="center")
            ax.set_axis_off()
        else:
            _counts, bin_edges = histogram
            ax.hist(diameters_um, bins=bin_edges, color="tab:blue", edgecolor="black")
            ax.set_xlabel("Partikeldurchmesser (µm)")
            ax.set_ylabel("Häufigkeit")
            ax.set_title("Durchmesserverteilung")
            if stats is not None:
                text_lines = [f"Anzahl: {stats.count}"]
                if stats.mean_diameter_um is not None:
                    text_lines.append(f"Mittelwert: {stats.mean_diameter_um:.3f} µm")
                if stats.std_diameter_um is not None:
                    text_lines.append(f"Std.-Abw.: {stats.std_diameter_um:.3f} µm")
                ax.text(
                    0.98,
                    0.95,
                    "\n".join(text_lines),
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
                )
            ax.grid(True, alpha=0.2)
        self._figure.tight_layout()
        self.draw_idle()


class ParameterPanel(QtWidgets.QGroupBox):
    """Einstellmöglichkeiten für die Analyseparameter."""

    parameters_changed = QtCore.Signal(dict)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__("Parameter", parent)
        form = QtWidgets.QFormLayout(self)

        self.gaussian_sigma = QtWidgets.QDoubleSpinBox()
        self.gaussian_sigma.setRange(0.0, 20.0)
        self.gaussian_sigma.setSingleStep(0.1)
        self.gaussian_sigma.setValue(2.0)
        form.addRow("Glättung σ", self.gaussian_sigma)

        self.edge_method = QtWidgets.QComboBox()
        self.edge_method.addItem("Canny", "canny")
        self.edge_method.addItem("Sobel", "sobel")
        self.edge_method.addItem("Scharr", "scharr")
        self.edge_method.addItem("Prewitt", "prewitt")
        form.addRow("Edge-Filter", self.edge_method)

        self.canny_sigma = QtWidgets.QDoubleSpinBox()
        self.canny_sigma.setRange(0.0, 10.0)
        self.canny_sigma.setSingleStep(0.1)
        self.canny_sigma.setValue(1.0)
        form.addRow("Canny σ", self.canny_sigma)

        self.canny_low = QtWidgets.QDoubleSpinBox()
        self.canny_low.setRange(0.0, 1.0)
        self.canny_low.setSingleStep(0.01)
        self.canny_low.setValue(0.1)
        form.addRow("Canny low", self.canny_low)

        self.canny_high = QtWidgets.QDoubleSpinBox()
        self.canny_high.setRange(0.0, 1.0)
        self.canny_high.setSingleStep(0.01)
        self.canny_high.setValue(0.3)
        form.addRow("Canny high", self.canny_high)

        self.edge_threshold = QtWidgets.QDoubleSpinBox()
        self.edge_threshold.setRange(-1.0, 5.0)
        self.edge_threshold.setSingleStep(0.05)
        self.edge_threshold.setValue(0.2)
        form.addRow("Gradient-Schwelle", self.edge_threshold)

        self.min_radius = QtWidgets.QSpinBox()
        self.min_radius.setRange(1, 1_000)
        self.min_radius.setValue(5)
        form.addRow("Min. Radius [px]", self.min_radius)

        self.max_radius = QtWidgets.QSpinBox()
        self.max_radius.setRange(1, 1_000)
        self.max_radius.setValue(50)
        form.addRow("Max. Radius [px]", self.max_radius)

        self.radius_step = QtWidgets.QSpinBox()
        self.radius_step.setRange(1, 200)
        self.radius_step.setValue(2)
        form.addRow("Radius-Schritt [px]", self.radius_step)

        self.total_peaks = QtWidgets.QSpinBox()
        self.total_peaks.setRange(1, 1_000)
        self.total_peaks.setValue(20)
        form.addRow("Max. Kreise", self.total_peaks)

        self.histogram_bins = QtWidgets.QSpinBox()
        self.histogram_bins.setRange(1, 200)
        self.histogram_bins.setValue(10)
        form.addRow("Histogramm-Bins", self.histogram_bins)

        for widget in (
            self.gaussian_sigma,
            self.edge_method,
            self.canny_sigma,
            self.canny_low,
            self.canny_high,
            self.edge_threshold,
            self.min_radius,
            self.max_radius,
            self.radius_step,
            self.total_peaks,
            self.histogram_bins,
        ):
            if isinstance(widget, QtWidgets.QComboBox):
                widget.currentIndexChanged.connect(self._emit_parameters)
            else:
                widget.valueChanged.connect(self._emit_parameters)

        self.canny_low.valueChanged.connect(self._synchronise_thresholds)
        self._update_enabled()
        self.edge_method.currentIndexChanged.connect(self._update_enabled)

    def parameters(self) -> Dict[str, float | int | str]:
        return {
            "gaussian_sigma": float(self.gaussian_sigma.value()),
            "edge_method": str(self.edge_method.currentData()),
            "canny_sigma": float(self.canny_sigma.value()),
            "canny_low": float(self.canny_low.value()),
            "canny_high": float(self.canny_high.value()),
            "edge_threshold": float(self.edge_threshold.value()),
            "min_radius": int(self.min_radius.value()),
            "max_radius": int(self.max_radius.value()),
            "radius_step": int(self.radius_step.value()),
            "total_peaks": int(self.total_peaks.value()),
            "histogram_bins": int(self.histogram_bins.value()),
        }

    def _emit_parameters(self) -> None:
        self.parameters_changed.emit(self.parameters())

    def _synchronise_thresholds(self, value: float) -> None:
        if self.canny_high.value() < value:
            self.canny_high.setValue(value)

    def _update_enabled(self) -> None:
        is_canny = self.edge_method.currentData() == "canny"
        for widget in (self.canny_sigma, self.canny_low, self.canny_high):
            widget.setEnabled(is_canny)
        self.edge_threshold.setEnabled(not is_canny)
        self._emit_parameters()


class SemParticleWindow(QtWidgets.QMainWindow):
    """Hauptfenster für die interaktive Analyse."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SEM Partikel Analyse")
        self.resize(1200, 700)

        self._image: Optional[np.ndarray] = None
        self._dpi: Optional[float] = None
        self._image_path: Optional[Path] = None

        self._param_panel = ParameterPanel()
        self._param_panel.parameters_changed.connect(self._on_parameters_changed)

        self._original_display = ImageDisplay("Originalbild")
        self._overlay_display = ImageDisplay("Overlay")
        self._edges_display = ImageDisplay("Kantenbild")

        self._tab_widget = QtWidgets.QTabWidget()
        self._tab_widget.addTab(self._original_display, "Original")
        self._tab_widget.addTab(self._overlay_display, "Overlay")
        self._tab_widget.addTab(self._edges_display, "Kanten")

        self._hist_canvas = HistogramCanvas()

        left_container = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_container)
        left_layout.addWidget(self._tab_widget, stretch=4)
        left_layout.addWidget(self._hist_canvas, stretch=3)

        self._stats_text = QtWidgets.QPlainTextEdit()
        self._stats_text.setReadOnly(True)

        right_container = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_container)
        right_layout.addWidget(self._param_panel)
        right_layout.addWidget(QtWidgets.QLabel("Statistik"))
        right_layout.addWidget(self._stats_text, 1)
        right_layout.addStretch(1)

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(left_container)
        splitter.addWidget(right_container)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        central = QtWidgets.QWidget()
        central_layout = QtWidgets.QHBoxLayout(central)
        central_layout.addWidget(splitter)
        self.setCentralWidget(central)

        self._create_actions()
        self.statusBar().showMessage("Bitte ein Bild laden…")

    # region UI initialisation helpers
    def _create_actions(self) -> None:
        file_menu = self.menuBar().addMenu("&Datei")

        open_action = QtWidgets.QAction("Bild öffnen…", self)
        open_action.setShortcut(QtGui.QKeySequence.Open)
        open_action.triggered.connect(self._open_image_dialog)
        file_menu.addAction(open_action)

        save_hist_action = QtWidgets.QAction("Histogramm speichern…", self)
        save_hist_action.triggered.connect(self._save_histogram)
        file_menu.addAction(save_hist_action)

        exit_action = QtWidgets.QAction("Beenden", self)
        exit_action.setShortcut(QtGui.QKeySequence.Quit)
        exit_action.triggered.connect(QtWidgets.QApplication.quit)
        file_menu.addAction(exit_action)

    # endregion

    def _open_image_dialog(self) -> None:
        start_dir = str(self._image_path.parent) if self._image_path else str(Path.home())
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "SEM-Bild öffnen",
            start_dir,
            "Bilder (*.tif *.tiff *.png *.jpg *.jpeg *.bmp *.gif);;Alle Dateien (*)",
        )
        if file_path:
            self.load_image(Path(file_path))

    def load_image(self, path: Path) -> None:
        try:
            image, dpi = load_image_with_dpi(path)
        except Exception as exc:  # pragma: no cover - defensive UI
            QtWidgets.QMessageBox.critical(self, "Fehler beim Laden", str(exc))
            return

        self._image = image
        self._dpi = dpi
        self._image_path = path
        self._original_display.set_array(image)
        self._overlay_display.set_array(None)
        self._edges_display.set_array(None)
        self._hist_canvas.update_histogram(None, None, None)
        self._stats_text.clear()
        msg = f"Geladen: {path.name}"
        if dpi:
            msg += f" – DPI: {dpi:.1f}"
        else:
            msg += " – keine DPI-Angabe"
        self.statusBar().showMessage(msg)
        self._run_analysis()

    def _on_parameters_changed(self, _params: dict) -> None:
        self._run_analysis()

    def _run_analysis(self) -> None:
        if self._image is None or self._image.size == 0:
            return

        params = self._param_panel.parameters()
        try:
            circles, diameters_um, histogram, stats, edges, microns_per_pixel = perform_analysis(
                self._image,
                self._dpi,
                gaussian_sigma=float(params["gaussian_sigma"]),
                edge_method=str(params["edge_method"]),
                canny_sigma=float(params["canny_sigma"]),
                canny_low=float(params["canny_low"]),
                canny_high=float(params["canny_high"]),
                edge_threshold=float(params["edge_threshold"]),
                min_radius=int(params["min_radius"]),
                max_radius=int(params["max_radius"]),
                radius_step=int(params["radius_step"]),
                total_peaks=int(params["total_peaks"]),
                histogram_bins=int(params["histogram_bins"]),
            )
        except Exception as exc:  # pragma: no cover - defensive UI
            self.statusBar().showMessage(f"Analysefehler: {exc}")
            return

        overlay = create_overlay(self._image, circles)
        self._overlay_display.set_array(overlay)
        self._edges_display.set_array(edges.astype(float))
        self._hist_canvas.update_histogram(diameters_um, histogram, stats)

        stats_lines = []
        if self._image_path is not None:
            stats_lines.append(f"Datei: {self._image_path.name}")
        stats_lines.append(f"Partikel gesamt: {stats.count}")
        if stats.mean_diameter_um is not None:
            stats_lines.append(f"Ø Durchmesser: {stats.mean_diameter_um:.3f} µm")
        if stats.std_diameter_um is not None:
            stats_lines.append(f"Std.-Abweichung: {stats.std_diameter_um:.3f} µm")
        if stats.mean_area_um2 is not None:
            stats_lines.append(f"Ø Fläche: {stats.mean_area_um2:.3f} µm²")
        if stats.std_area_um2 is not None:
            stats_lines.append(f"Std.-Abweichung Fläche: {stats.std_area_um2:.3f} µm²")
        if microns_per_pixel is not None:
            stats_lines.append(f"Mikrometer pro Pixel: {microns_per_pixel:.6f} µm")
        else:
            stats_lines.append("Mikrometer pro Pixel: nicht verfügbar")

        if diameters_um is not None:
            stats_lines.append(f"Durchmesser (min/max): {diameters_um.min():.3f} / {diameters_um.max():.3f} µm")

        self._stats_text.setPlainText("\n".join(stats_lines))
        self.statusBar().showMessage("Analyse abgeschlossen")

    def _save_histogram(self) -> None:
        if self._hist_canvas.figure.axes and self._hist_canvas.figure.axes[0].has_data():
            file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Histogramm speichern",
                str(self._image_path.parent) if self._image_path else str(Path.home()),
                "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)",
            )
            if file_path:
                self._hist_canvas.figure.savefig(file_path, dpi=300)
                self.statusBar().showMessage(f"Histogramm gespeichert unter: {file_path}")
        else:
            QtWidgets.QMessageBox.information(self, "Keine Daten", "Es gibt kein Histogramm zum Speichern.")


def main(argv: Sequence[str] | None = None) -> int:
    """Qt-Anwendung starten."""

    if argv is None:
        argv = sys.argv[1:]

    app = QtWidgets.QApplication.instance()
    if app is None:
        qt_args = ["sem-particle-viewer", *argv]
        app = QtWidgets.QApplication(qt_args)

    window = SemParticleWindow()
    if argv:
        first_path = Path(argv[0])
        if first_path.exists():
            window.load_image(first_path)
        else:
            QtWidgets.QMessageBox.warning(window, "Datei nicht gefunden", str(first_path))
    window.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover - Qt entry point
    raise SystemExit(main())
