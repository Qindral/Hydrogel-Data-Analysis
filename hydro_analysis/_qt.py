"""Qt widgets for Hydro Analysis plugin."""
from __future__ import annotations

from typing import Iterable, List, Optional

from qtpy import QtCore, QtWidgets

from .metadata import DatasetMetadata, format_stage_position, format_timestamp_summary


class InfoPanel(QtWidgets.QWidget):
    """Read-only metadata display."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent=parent)
        self._fields: List[QtWidgets.QLabel] = []
        layout = QtWidgets.QFormLayout(self)
        layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self._labels = {
            "core": QtWidgets.QLabel("–"),
            "cal": QtWidgets.QLabel("–"),
            "stage": QtWidgets.QLabel("–"),
            "kind": QtWidgets.QLabel("–"),
            "axes": QtWidgets.QLabel("–"),
        }
        for title, label in (
            ("Core", self._labels["core"]),
            ("Axes", self._labels["axes"]),
            ("Calibration", self._labels["cal"]),
            ("Stage", self._labels["stage"]),
            ("Kind", self._labels["kind"]),
        ):
            label.setWordWrap(True)
            layout.addRow(title + ":", label)
        self.setLayout(layout)

    def update_metadata(self, metadata: DatasetMetadata) -> None:
        width, height = metadata.width_height
        dtype = metadata.dtype
        counts = metadata.counts
        core = f"{width}×{height} px, {dtype}, C={counts.get('C', 1)}, Z={counts.get('Z', 1)}, T={counts.get('T', 1)}"
        self._labels["core"].setText(core)
        self._labels["axes"].setText(metadata.axes)
        cal_parts: List[str] = []
        if metadata.px_size_xy_um:
            cal_parts.append(f"px={metadata.px_size_xy_um:.3f} µm")
        if metadata.z_step_um:
            cal_parts.append(f"Δz={metadata.z_step_um:.3f} µm")
        cal_summary = format_timestamp_summary(metadata.timestamps)
        if cal_summary != "–":
            cal_parts.append(cal_summary)
        self._labels["cal"].setText(", ".join(cal_parts) if cal_parts else "–")
        self._labels["stage"].setText(format_stage_position(metadata.stage_position_mm))
        self._labels["kind"].setText(metadata.infer_kind())


class DisplayPanel(QtWidgets.QWidget):
    """Display controls for the viewer."""

    colormap_changed = QtCore.Signal(str)
    sigma_changed = QtCore.Signal(float)
    filtered_visible_changed = QtCore.Signal(bool)
    scalebar_toggled = QtCore.Signal(bool)

    def __init__(self, available_colormaps: Iterable[str], parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent=parent)
        layout = QtWidgets.QVBoxLayout(self)
        self.scalebar_checkbox = QtWidgets.QCheckBox("Scalebar anzeigen")
        layout.addWidget(self.scalebar_checkbox)
        cmap_layout = QtWidgets.QHBoxLayout()
        cmap_layout.addWidget(QtWidgets.QLabel("Colormap"))
        self.colormap_combo = QtWidgets.QComboBox()
        self.colormap_combo.addItems(sorted(set(available_colormaps)))
        cmap_layout.addWidget(self.colormap_combo, 1)
        layout.addLayout(cmap_layout)
        sigma_layout = QtWidgets.QHBoxLayout()
        sigma_layout.addWidget(QtWidgets.QLabel("σ [px]"))
        self.sigma_spin = QtWidgets.QDoubleSpinBox()
        self.sigma_spin.setRange(0.0, 3.0)
        self.sigma_spin.setSingleStep(0.1)
        self.sigma_spin.setValue(0.0)
        sigma_layout.addWidget(self.sigma_spin, 1)
        layout.addLayout(sigma_layout)
        self.filtered_checkbox = QtWidgets.QCheckBox("Filtered anzeigen")
        layout.addWidget(self.filtered_checkbox)
        layout.addStretch(1)
        self.setLayout(layout)
        self.scalebar_checkbox.stateChanged.connect(lambda state: self.scalebar_toggled.emit(state == QtCore.Qt.Checked))
        self.colormap_combo.currentTextChanged.connect(self.colormap_changed.emit)
        self.sigma_spin.valueChanged.connect(self.sigma_changed.emit)
        self.filtered_checkbox.stateChanged.connect(lambda state: self.filtered_visible_changed.emit(state == QtCore.Qt.Checked))

    def set_state(self, *, scalebar: bool, colormap: str, sigma: float, filtered_visible: bool) -> None:
        self.scalebar_checkbox.blockSignals(True)
        self.scalebar_checkbox.setChecked(scalebar)
        self.scalebar_checkbox.blockSignals(False)

        self.colormap_combo.blockSignals(True)
        index = self.colormap_combo.findText(colormap)
        if index >= 0:
            self.colormap_combo.setCurrentIndex(index)
        self.colormap_combo.blockSignals(False)

        self.sigma_spin.blockSignals(True)
        self.sigma_spin.setValue(sigma)
        self.sigma_spin.blockSignals(False)

        self.filtered_checkbox.blockSignals(True)
        self.filtered_checkbox.setChecked(filtered_visible)
        self.filtered_checkbox.blockSignals(False)
