import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6 import QtWidgets  # noqa: E402

from ewald import Pipeline, StaticSource  # noqa: E402
from ewald import modifiers as mods  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def pipe(cu_fcc):
    return Pipeline(StaticSource([cu_fcc, cu_fcc]))


def test_pipeline_panel_add_toggle_reorder(app, pipe):
    from ewald.gui.pipeline_panel import PipelinePanel

    panel = PipelinePanel()
    panel.set_pipeline(pipe)
    changes = []
    panel.changed.connect(lambda: changes.append(1))
    panel.add_modifier(mods.CoordinationAnalysis(3.0))
    panel.add_modifier(mods.ColorCoding("Coordination"))
    assert [m.name for m in pipe.modifiers] == ["CoordinationAnalysis", "ColorCoding"]
    assert panel.list.item(0).text() == "ColorCoding"  # newest on top, like OVITO
    panel.list.setCurrentRow(0)
    panel._move(1)
    assert [m.name for m in pipe.modifiers] == ["ColorCoding", "CoordinationAnalysis"]
    panel.list.item(0).setCheckState(panel.list.item(0).checkState().__class__.Unchecked)
    assert pipe.modifiers[1].enabled is False
    assert len(changes) >= 3


def test_parameter_form_roundtrip(app):
    from ewald.gui.params import ParameterForm

    mod = mods.Slice(normal=(0, 0, 1), distance=1.0)
    form = ParameterForm(mod)
    form.editors["distance"][0].setValue(2.5)
    form.editors["normal"][0].setText("(1, 0, 0)")
    form.editors["inverse"][0].setChecked(True)
    form.apply()
    assert mod.distance == 2.5 and mod.normal == (1, 0, 0) and mod.inverse is True


def test_python_script_modifier_and_error_skip(pipe):
    script = mods.PythonScript("def modify(frame, data):\n    data['F'] = np.full(data.count, frame)\n")
    bad = mods.ExpressionSelection("NoSuchProperty > 1")
    pipe.modifiers.extend([script, bad])
    d = pipe.compute(1, errors="skip")
    assert np.all(d["F"] == 1)
    assert bad.status and "NoSuchProperty" in bad.status
    with pytest.raises(NameError):
        pipe.compute(1)


def test_inspector_filter(app, cu_fcc):
    from ewald.gui.inspector import DataInspector

    insp = DataInspector()
    insp.set_data(cu_fcc)
    insp.filter.setText("Position.Z < 1")
    insp._apply_filter()
    assert insp.model.rowCount() == int((cu_fcc.positions[:, 2] < 1).sum())


def test_timeline(app):
    from ewald.gui.timeline import Timeline

    tl = Timeline()
    seen = []
    tl.frameChanged.connect(seen.append)
    tl.set_num_frames(5)
    tl._set(3)
    tl._advance()
    tl._advance()
    assert seen == [3, 4, 0]


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="main window needs an X display with OpenGL (xvfb-run)")
def test_main_window_smoke(app, pipe, tmp_path):
    from ewald.gui.app import MainWindow

    w = MainWindow(pipe)
    w.panel.add_modifier(mods.CoordinationAnalysis(3.0))
    w.timeline._set(1)
    assert w.data.count == pipe.compute(1).count and "Coordination" in w.data
    out = w.render_image(str(tmp_path / "x.png"))
    assert os.path.getsize(out) > 1000
    w.close()
