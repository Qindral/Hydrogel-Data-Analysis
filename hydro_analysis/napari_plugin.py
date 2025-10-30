"""Napari plugin entry points for Hydro Analysis."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from napari.utils import notifications
from napari.utils.colormaps import AVAILABLE_COLORMAPS
from napari.qt.threading import thread_worker
from qtpy import QtWidgets
from scipy.ndimage import gaussian_filter

from ._qt import DisplayPanel, InfoPanel
from .data_loader import DatasetLoader
from .metadata import DatasetMetadata, ensure_dataset_root
from .spt_widget import SPTPreprocessingWidget

RAW_LAYER_NAME = "Raw"
FILTERED_LAYER_NAME = "Filtered"


def _get_or_create_dock(viewer, widget, name: str, area="right"):
    for dock in viewer.window._dock_widgets.values():
        if dock.widget() is widget:
            return dock
    dock = viewer.window.add_dock_widget(widget, area=area, name=name)
    dock.setObjectName(name)
    return dock


def _ensure_layers(viewer):
    raw_layer = viewer.layers[RAW_LAYER_NAME] if RAW_LAYER_NAME in viewer.layers else None
    filtered_layer = viewer.layers[FILTERED_LAYER_NAME] if FILTERED_LAYER_NAME in viewer.layers else None
    return raw_layer, filtered_layer


def _update_scale(layer, metadata: DatasetMetadata) -> None:
    axes = metadata.axes
    scale = [1.0] * layer.data.ndim
    if "Z" in axes and metadata.z_step_um:
        scale[axes.index("Z")] = metadata.z_step_um
    if metadata.px_size_xy_um:
        if "Y" in axes:
            scale[axes.index("Y")] = metadata.px_size_xy_um
        if "X" in axes:
            scale[axes.index("X")] = metadata.px_size_xy_um
    layer.scale = tuple(scale)


def open_dataset_dialog(viewer) -> None:
    dialog = QtWidgets.QFileDialog(viewer.window.qt_viewer)
    dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
    dialog.setNameFilters(["TIFF files (*.tif *.tiff)"])
    dialog.setWindowTitle("Open Dataset…")
    if not dialog.exec_():
        return
    paths = dialog.selectedFiles()
    if not paths:
        return
    path = Path(paths[0])
    try:
        loader = DatasetLoader(path)
        loaded = loader.load()
    except Exception as exc:  # noqa: BLE001
        notifications.show_error(f"Fehler beim Laden: {exc}")
        return
    data = loaded.data
    metadata = loaded.metadata
    raw_layer, filtered_layer = _ensure_layers(viewer)
    if raw_layer is None:
        raw_layer = viewer.add_image(
            data,
            name=RAW_LAYER_NAME,
            multiscale=False,
            blending="translucent",
            metadata={"dataset_path": str(path), "metadata": metadata},
        )
    else:
        raw_layer.data = data
        raw_layer.metadata = {"dataset_path": str(path), "metadata": metadata}
    _update_scale(raw_layer, metadata)
    raw_layer.colormap = "gray"
    raw_layer.visible = True
    if filtered_layer is None:
        filtered_layer = viewer.add_image(
            data,
            name=FILTERED_LAYER_NAME,
            visible=False,
            blending="additive",
            colormap="magma",
            metadata={"derived_from": RAW_LAYER_NAME},
        )
    else:
        filtered_layer.data = data
    _update_scale(filtered_layer, metadata)
    info_panel = getattr(viewer.window, "_hydro_info_panel", None)
    if info_panel is None:
        info_panel = InfoPanel()
        viewer.window._hydro_info_panel = info_panel
    info_panel.update_metadata(metadata)
    _get_or_create_dock(viewer, info_panel, "Info")
    display_panel = getattr(viewer.window, "_hydro_display_panel", None)
    if display_panel is None:
        display_panel = DisplayPanel(AVAILABLE_COLORMAPS.keys())
        viewer.window._hydro_display_panel = display_panel
        display_panel.colormap_changed.connect(lambda name: _set_colormap(viewer, name))
        display_panel.sigma_changed.connect(lambda sigma: _update_filtered(viewer, sigma))
        display_panel.filtered_visible_changed.connect(lambda state: _toggle_filtered(viewer, state))
        display_panel.scalebar_toggled.connect(lambda state: setattr(viewer.scale_bar, "visible", state))
    display_panel.set_state(
        scalebar=viewer.scale_bar.visible,
        colormap=raw_layer.colormap.name if hasattr(raw_layer.colormap, "name") else raw_layer.colormap,
        sigma=0.0,
        filtered_visible=filtered_layer.visible,
    )
    _get_or_create_dock(viewer, display_panel, "Display")
    viewer.scale_bar.visible = display_panel.scalebar_checkbox.isChecked()
    viewer.scale_bar.unit = "µm"
    raw_layer.metadata["metadata"] = metadata
    _toggle_filtered(viewer, display_panel.filtered_checkbox.isChecked())
    notifications.show_info(f"Geladen: {path.name}")


def _set_colormap(viewer, name: str) -> None:
    if RAW_LAYER_NAME not in viewer.layers:
        return
    viewer.layers[RAW_LAYER_NAME].colormap = name


def _toggle_filtered(viewer, state: bool) -> None:
    if FILTERED_LAYER_NAME not in viewer.layers:
        return
    viewer.layers[FILTERED_LAYER_NAME].visible = state


def _update_filtered(viewer, sigma: float) -> None:
    if FILTERED_LAYER_NAME not in viewer.layers or RAW_LAYER_NAME not in viewer.layers:
        return
    filtered_layer = viewer.layers[FILTERED_LAYER_NAME]
    raw_layer = viewer.layers[RAW_LAYER_NAME]
    if sigma <= 0:
        filtered_layer.data = raw_layer.data
        return

    metadata: DatasetMetadata = raw_layer.metadata.get("metadata")
    axes = metadata.axes if metadata else "" * raw_layer.data.ndim

    @thread_worker
    def _worker(data: np.ndarray, sigma_value: float):
        sigma_arr = [sigma_value if axis in ("Z", "Y", "X") else 0 for axis in axes]
        if len(sigma_arr) != data.ndim:
            sigma_arr = [sigma_value if i >= data.ndim - 3 else 0 for i in range(data.ndim)]
        return gaussian_filter(data, sigma=sigma_arr)

    def _on_result(result):
        filtered_layer.data = result

    worker = _worker(raw_layer.data, sigma)
    worker.returned.connect(_on_result)
    worker.start()


def save_metadata_command(viewer) -> None:
    raw_layer = viewer.layers.get(RAW_LAYER_NAME)
    if raw_layer is None:
        notifications.show_error("Kein Dataset geladen.")
        return
    metadata: Optional[DatasetMetadata] = raw_layer.metadata.get("metadata")
    if metadata is None:
        notifications.show_error("Keine Metadaten verfügbar.")
        return
    root = ensure_dataset_root(metadata.path)
    meta_path = root / "meta.json"
    try:
        metadata.to_json(meta_path)
    except Exception as exc:  # noqa: BLE001
        notifications.show_error(f"Fehler beim Speichern: {exc}")
        return
    notifications.show_info(f"Metadaten gespeichert: {meta_path}")


def launch_spt_preprocessing(viewer) -> None:
    widget = getattr(viewer.window, "_hydro_spt_widget", None)
    if widget is None:
        widget = SPTPreprocessingWidget(viewer)
        viewer.window._hydro_spt_widget = widget
    dock = _get_or_create_dock(viewer, widget, "SPT Preprocessing")
    dock.show()
    dock.raise_()
