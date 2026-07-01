import os
import sys
import importlib.util

import pytest

# Ensure repo root is on path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def test_all_phase_scripts_import():
    """Every script in code/ should import without syntax errors."""
    scripts = [
        "code/p0_logit_lens/logit_lens.py",
        "code/p2_probes/p2_linear_probes.py",
        "code/p3_translator/p3_mlp_translator.py",
        "code/p4_steering/p4_steering.py",
    ]
    for script in scripts:
        path = os.path.join(REPO_ROOT, script)
        if not os.path.exists(path):
            pytest.skip(f"{script} not found")
        spec = importlib.util.spec_from_file_location("phase_script", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)


def test_demo_app_imports():
    """The Gradio demo script should import cleanly."""
    demo_path = os.path.join(REPO_ROOT, "demo", "app.py")
    if not os.path.exists(demo_path):
        pytest.skip("demo/app.py not found")
    spec = importlib.util.spec_from_file_location("demo_app", demo_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def test_data_files_exist():
    """Core generated artifacts should be present in data/."""
    required = [
        "data/probe_results.pkl",
        "data/p3_metrics.pkl",
        "data/p4_steering_results.pkl",
    ]
    for f in required:
        path = os.path.join(REPO_ROOT, f)
        if not os.path.exists(path):
            pytest.skip(f"{f} not found (expected after running experiments)")


def test_figures_exist():
    """Key figures should be present in figures/."""
    required = [
        "figures/p2_f1_by_layer.png",
        "figures/p3_mlp_vs_linear.png",
        "figures/p4_steering_results.png",
    ]
    for f in required:
        path = os.path.join(REPO_ROOT, f)
        assert os.path.exists(path), f"Missing figure: {f}"
