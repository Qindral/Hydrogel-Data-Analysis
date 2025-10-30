"""Qt widget for interactive SPT preprocessing."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from napari.qt.threading import thread_worker
from napari.utils import notifications
from qtpy import QtWidgets

from .data_loader import DatasetLoader
from .metadata import DatasetMetadata
from .spt_processing import (
    SPTPreprocessingParams,
    SPTPreprocessingResult,
    run_spt_preprocessing,
)

RAW_LAYER = "SPT Raw"
VAR_LAYER = "SPT Variance"
Z_LAYER = "SPT Z-Score"
BANDPASS_LAYER = "SPT Bandpass"
PEAKS_LAYER = "SPT Peaks"


class SPTPreprocessingWidget(QtWidgets.QWidget):
    """Interactive widget that runs the SPT preprocessing pipeline."""

    def __init__(self, viewer) -> None:
        super().__init__()
        self.viewer = viewer
        self._video: Optional[np.ndarray] = None
        self._metadata: Optional[DatasetMetadata] = None
        self._worker = None
        self._build_ui()
        self._refresh_mask_choices()
        self.viewer.layers.events.inserted.connect(self._refresh_mask_choices)
        self.viewer.layers.events.removed.connect(self._refresh_mask_choices)

    # UI construction -------------------------------------------------
    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        file_layout = QtWidgets.QHBoxLayout()
        self.path_label = QtWidgets.QLabel("Kein Video geladen")
        self.path_label.setWordWrap(True)
        load_button = QtWidgets.QPushButton("TIFF laden…")
        load_button.clicked.connect(self._on_load_clicked)
        file_layout.addWidget(self.path_label, 1)
        file_layout.addWidget(load_button)
        layout.addLayout(file_layout)

        param_group = QtWidgets.QGroupBox("Parameter")
        form = QtWidgets.QFormLayout(param_group)

        self.window_spin = QtWidgets.QSpinBox()
        self.window_spin.setRange(3, 999)
        self.window_spin.setValue(21)
        self.window_spin.setSingleStep(2)
        form.addRow("Fenster N", self.window_spin)

        self.log_sigma_spin = QtWidgets.QDoubleSpinBox()
        self.log_sigma_spin.setRange(0.0, 10.0)
        self.log_sigma_spin.setDecimals(2)
        self.log_sigma_spin.setSingleStep(0.1)
        self.log_sigma_spin.setValue(1.5)
        form.addRow("σ (LoG)", self.log_sigma_spin)

        self.threshold_spin = QtWidgets.QDoubleSpinBox()
        self.threshold_spin.setRange(-10.0, 1000.0)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setSingleStep(0.1)
        self.threshold_spin.setValue(1.5)
        form.addRow("Schwelle T", self.threshold_spin)

        self.nms_radius_spin = QtWidgets.QDoubleSpinBox()
        self.nms_radius_spin.setRange(1.0, 50.0)
        self.nms_radius_spin.setDecimals(1)
        self.nms_radius_spin.setValue(3.0)
        form.addRow("NMS Radius", self.nms_radius_spin)

        self.zscore_checkbox = QtWidgets.QCheckBox("Z-Score nutzen")
        self.zscore_checkbox.setChecked(True)
        form.addRow(self.zscore_checkbox)

        self.drift_checkbox = QtWidgets.QCheckBox("Driftkorrektur aktivieren")
        self.drift_checkbox.setChecked(False)
        form.addRow(self.drift_checkbox)

        self.drift_interval_spin = QtWidgets.QSpinBox()
        self.drift_interval_spin.setRange(1, 999)
        self.drift_interval_spin.setValue(20)
        form.addRow("Drift Update", self.drift_interval_spin)

        self.bleach_combo = QtWidgets.QComboBox()
        self.bleach_combo.addItems(["none", "poly", "exp"])
        form.addRow("Bleaching", self.bleach_combo)

        self.bleach_order_spin = QtWidgets.QSpinBox()
        self.bleach_order_spin.setRange(1, 5)
        self.bleach_order_spin.setValue(1)
        form.addRow("Bleach Ordnung", self.bleach_order_spin)

        self.presence_fraction_spin = QtWidgets.QDoubleSpinBox()
        self.presence_fraction_spin.setRange(0.0, 1.0)
        self.presence_fraction_spin.setSingleStep(0.05)
        self.presence_fraction_spin.setValue(0.0)
        form.addRow("Min Präsenz", self.presence_fraction_spin)

        self.presence_threshold_spin = QtWidgets.QDoubleSpinBox()
        self.presence_threshold_spin.setRange(0.0, 10000.0)
        self.presence_threshold_spin.setSingleStep(0.5)
        self.presence_threshold_spin.setValue(0.0)
        form.addRow("Präsenz Schwelle", self.presence_threshold_spin)

        layout.addWidget(param_group)

        mask_layout = QtWidgets.QHBoxLayout()
        mask_layout.addWidget(QtWidgets.QLabel("ROI/Mask"))
        self.mask_combo = QtWidgets.QComboBox()
        mask_layout.addWidget(self.mask_combo, 1)
        layout.addLayout(mask_layout)

        run_button = QtWidgets.QPushButton("Pipeline ausführen")
        run_button.clicked.connect(self._run_pipeline)
        layout.addWidget(run_button)

        self.summary_box = QtWidgets.QPlainTextEdit()
        self.summary_box.setReadOnly(True)
        self.summary_box.setPlaceholderText("Ergebnisse & QS Kennzahlen erscheinen hier…")
        layout.addWidget(self.summary_box, 1)

        self.setLayout(layout)

    # Dataset loading -------------------------------------------------
    def _on_load_clicked(self) -> None:
        dialog = QtWidgets.QFileDialog(self)
        dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        dialog.setNameFilters(["TIFF files (*.tif *.tiff)"])
        dialog.setWindowTitle("SPT Dataset öffnen…")
        if not dialog.exec_():
            return
        files = dialog.selectedFiles()
        if not files:
            return
        path = Path(files[0])
        try:
            loader = DatasetLoader(path)
            loaded = loader.load()
        except Exception as exc:  # noqa: BLE001
            notifications.show_error(f"Fehler beim Laden: {exc}")
            return
        try:
            video = self._prepare_video(loaded.data, loaded.metadata)
        except ValueError as exc:
            notifications.show_error(str(exc))
            return
        self._video = video
        self._metadata = loaded.metadata
        self.path_label.setText(str(path))
        self.summary_box.setPlainText(
            f"Geladen: {path.name}\nForm: {video.shape} (T,Y,X)\nMetadaten Achsen: {loaded.metadata.axes}"
        )
        self._push_raw_layer(video)
        notifications.show_info(f"Dataset geladen: {path.name}")

    @staticmethod
    def _prepare_video(data: np.ndarray, metadata: DatasetMetadata) -> np.ndarray:
        axes = metadata.axes
        array = np.asarray(data)
        working_axes = list(axes)
        working_array = array
        if "C" in working_axes:
            c_axis = working_axes.index("C")
            working_array = np.take(working_array, 0, axis=c_axis)
            working_axes.pop(c_axis)
        if "Z" in working_axes:
            z_axis = working_axes.index("Z")
            working_array = working_array.max(axis=z_axis)
            working_axes.pop(z_axis)
        if "T" not in working_axes:
            raise ValueError("Dataset besitzt keine Zeitachse (T)")
        # Reorder to T, Y, X
        perm = [working_axes.index(axis) for axis in ("T", "Y", "X") if axis in working_axes]
        if len(perm) != 3:
            raise ValueError("Dataset benötigt T/Y/X Achsen")
        working_array = np.moveaxis(working_array, perm, (0, 1, 2))
        # Ensure contiguous (T, Y, X)
        if working_array.ndim != 3:
            raise ValueError(f"Dataset muss 3D (T,Y,X) sein, erhalten {working_array.shape}")
        working_array = np.ascontiguousarray(working_array)
        return working_array.astype(np.float32)

    # Mask handling ---------------------------------------------------
    def _refresh_mask_choices(self, *args, **kwargs) -> None:  # noqa: D401
        current = self.mask_combo.currentText() if self.mask_combo.count() else ""
        self.mask_combo.blockSignals(True)
        self.mask_combo.clear()
        self.mask_combo.addItem("(keine)")
        for layer in self.viewer.layers:
            if layer.name in (RAW_LAYER, VAR_LAYER, Z_LAYER, BANDPASS_LAYER, PEAKS_LAYER):
                continue
            if getattr(layer, "ndim", 0) in (2, 3):
                self.mask_combo.addItem(layer.name)
        idx = self.mask_combo.findText(current)
        if idx >= 0:
            self.mask_combo.setCurrentIndex(idx)
        self.mask_combo.blockSignals(False)

    def _selected_mask(self) -> Optional[np.ndarray]:
        name = self.mask_combo.currentText()
        if not name or name == "(keine)":
            return None
        if name not in self.viewer.layers:
            return None
        layer = self.viewer.layers[name]
        data = np.asarray(layer.data)
        if data.ndim == 2:
            mask = data > 0
        elif data.ndim == 3:
            # If the mask includes time, collapse along axis 0
            mask = data.max(axis=0) > 0
        else:
            notifications.show_warning("Maskenlayer muss 2D oder 3D sein")
            return None
        return mask.astype(bool)

    # Pipeline execution ----------------------------------------------
    def _run_pipeline(self) -> None:
        if self._video is None:
            notifications.show_error("Bitte zuerst ein Video laden.")
            return
        params = SPTPreprocessingParams(
            window=self.window_spin.value(),
            log_sigma=self.log_sigma_spin.value(),
            threshold=self.threshold_spin.value(),
            nms_radius=self.nms_radius_spin.value(),
            use_zscore=self.zscore_checkbox.isChecked(),
            drift_correction=self.drift_checkbox.isChecked(),
            drift_update_interval=self.drift_interval_spin.value(),
            bleaching_mode=self.bleach_combo.currentText(),
            bleaching_order=self.bleach_order_spin.value(),
            min_presence_fraction=self.presence_fraction_spin.value(),
            presence_threshold=self.presence_threshold_spin.value(),
        )
        mask = self._selected_mask()

        @thread_worker
        def _worker():
            return run_spt_preprocessing(self._video, params, mask=mask)

        def _on_result(result: SPTPreprocessingResult) -> None:
            self._apply_result(result)
            notifications.show_info(f"{result.peaks.shape[0]} Peaks gefunden")

        def _on_error(err: BaseException) -> None:
            notifications.show_error(f"Pipeline Fehler: {err}")

        self.summary_box.setPlainText("Berechne…")
        worker = _worker()
        worker.returned.connect(_on_result)
        worker.errored.connect(_on_error)
        worker.start()
        self._worker = worker

    def _apply_result(self, result: SPTPreprocessingResult) -> None:
        self._push_raw_layer(result.corrected_video)
        self._push_map_layer(VAR_LAYER, result.variance_map)
        self._push_map_layer(Z_LAYER, result.zscore_map)
        self._push_map_layer(BANDPASS_LAYER, result.bandpass_map)
        self._push_points_layer(result.peaks)
        summary_lines = [
            f"Peaks: {result.peaks.shape[0]}",
            f"Var Mittel: {float(result.variance_map.mean()):.3f}",
            f"Bandpass max: {float(result.bandpass_map.max()):.3f}",
        ]
        if result.drift_shifts is not None:
            max_shift = max(np.hypot(y, x) for y, x in result.drift_shifts)
            summary_lines.append(f"Max Drift: {max_shift:.2f} px")
        if result.peak_metrics:
            mean_resp = float(np.mean([m.get("response", 0.0) for m in result.peak_metrics]))
            summary_lines.append(f"⟨Response⟩: {mean_resp:.3f}")
            if any("presence_fraction" in m for m in result.peak_metrics):
                mean_presence = float(np.mean([m.get("presence_fraction", 0.0) for m in result.peak_metrics]))
                summary_lines.append(f"⟨Präsenz⟩: {mean_presence:.2f}")
        self.summary_box.setPlainText("\n".join(summary_lines))

    # Layer helpers ---------------------------------------------------
    def _push_raw_layer(self, video: np.ndarray) -> None:
        if RAW_LAYER in self.viewer.layers:
            layer = self.viewer.layers[RAW_LAYER]
            layer.data = video
            layer.visible = True
        else:
            self.viewer.add_image(
                video,
                name=RAW_LAYER,
                blending="additive",
                colormap="gray",
                metadata={"kind": "SPT"},
            )

    def _push_map_layer(self, name: str, image: np.ndarray) -> None:
        if name in self.viewer.layers:
            layer = self.viewer.layers[name]
            layer.data = image
        else:
            self.viewer.add_image(
                image,
                name=name,
                blending="additive",
                colormap="magma" if name != Z_LAYER else "viridis",
                visible=name == BANDPASS_LAYER,
            )

    def _push_points_layer(self, coords: np.ndarray) -> None:
        if PEAKS_LAYER in self.viewer.layers:
            layer = self.viewer.layers[PEAKS_LAYER]
            layer.data = coords[:, ::-1] if coords.size else coords
        else:
            self.viewer.add_points(
                coords[:, ::-1] if coords.size else coords,
                name=PEAKS_LAYER,
                size=4,
                face_color="cyan",
                edge_color="black",
            )

    def deleteLater(self) -> None:  # noqa: D401
        if self._worker is not None:
            try:
                self._worker.quit()
            except Exception:  # pragma: no cover - best effort cleanup
                pass
        super().deleteLater()
