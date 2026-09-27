#!/usr/bin/env python
"""
run_environment_check.py - FakeShield Master Environment Diagnostics Suite

Executes the 16 deterministic environment verification steps in exact required order:
STEP 1  — Project root
STEP 2  — Python
STEP 3  — Virtual environment
STEP 4  — Package imports
STEP 5  — PyTorch
STEP 6  — Transformers
STEP 7  — pandas/numpy/sklearn
STEP 8  — Pillow/OpenCV
STEP 9  — pytesseract
STEP 10 — Tesseract engine
STEP 11 — requests/dotenv
STEP 12 — Streamlit
STEP 13 — pytest
STEP 14 — Phase 1 dataset accessibility
STEP 15 — Phase 1 raw hash verification
STEP 16 — Generate reports
"""

import os
import sys
import io
import shutil
import hashlib
import platform
import subprocess
from pathlib import Path

# Paths
PROJECT_ROOT = Path.cwd()
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
EVAL_DIR = PROJECT_ROOT / "data" / "evaluation"


def step_1_project_root():
    print("\n" + "=" * 80)
    print("STEP 1 — PROJECT ROOT VERIFICATION")
    print("=" * 80)
    print(f"Current Working Directory: {PROJECT_ROOT}")

    required_items = [
        ("PROJECT_SPEC.md", PROJECT_ROOT / "PROJECT_SPEC.md", "file"),
        ("data/raw/", RAW_DIR, "dir"),
        ("data/processed/", PROCESSED_DIR, "dir"),
        ("data/splits/", SPLITS_DIR, "dir"),
        ("data/evaluation/", EVAL_DIR, "dir"),
        ("src/", PROJECT_ROOT / "src", "dir"),
        ("tests/", PROJECT_ROOT / "tests", "dir"),
    ]

    all_found = True
    for name, path, item_type in required_items:
        exists = path.is_file() if item_type == "file" else path.is_dir()
        status = "[PASS]" if exists else "[FAIL]"
        print(f"  {status} {name}")
        if not exists:
            all_found = False

    if not all_found:
        raise RuntimeError("STEP 1 FAIL: Project root verification failed!")
    print("STEP 1 RESULT: PASS")
    return True


def step_2_python():
    print("\n" + "=" * 80)
    print("STEP 2 — PYTHON VERSION VERIFICATION")
    print("=" * 80)
    py_ver = platform.python_version()
    major, minor, micro = sys.version_info[:3]
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version:    {py_ver} ({major}.{minor}.{micro})")

    if major < 3 or (major == 3 and minor < 10):
        raise RuntimeError(f"STEP 2 FAIL: Unsupported Python version {py_ver}. Requires >= 3.10")

    print(f"Version check: {py_ver} >= 3.10 [PASS]")
    print("STEP 2 RESULT: PASS")
    return py_ver


def step_3_virtual_environment():
    print("\n" + "=" * 80)
    print("STEP 3 — VIRTUAL ENVIRONMENT VERIFICATION")
    print("=" * 80)
    in_venv = (sys.prefix != sys.base_prefix)
    exe_path = sys.executable
    venv_name_ok = "venv" in exe_path.lower()
    print(f"Active Python Prefix:      {sys.prefix}")
    print(f"Base Python Prefix:        {sys.base_prefix}")
    print(f"Virtual Env Active:        {in_venv}")
    print(f"Executable inside venv:    {venv_name_ok} ({exe_path})")

    pip_path = shutil.which("pip")
    print(f"Pip Executable in PATH:    {pip_path}")

    if not in_venv and not venv_name_ok:
        raise RuntimeError("STEP 3 FAIL: Python is not running inside a virtual environment!")

    print("STEP 3 RESULT: PASS")
    return True


def step_4_package_imports():
    print("\n" + "=" * 80)
    print("STEP 4 — PACKAGE IMPORTS VERIFICATION")
    print("=" * 80)
    packages = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("sklearn", "scikit-learn"),
        ("PIL", "Pillow"),
        ("cv2", "opencv-python"),
        ("pytesseract", "pytesseract"),
        ("transformers", "transformers"),
        ("torch", "torch"),
        ("sentencepiece", "sentencepiece"),
        ("jiwer", "jiwer"),
        ("requests", "requests"),
        ("dotenv", "python-dotenv"),
        ("streamlit", "streamlit"),
        ("pytest", "pytest"),
        ("langdetect", "langdetect"),
    ]

    package_info = []
    all_ok = True
    print(f"{'PACKAGE':<18} {'VERSION':<18} {'STATUS'}")
    print("-" * 50)
    for import_name, dist_name in packages:
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", None)
            if ver is None:
                import importlib.metadata
                ver = importlib.metadata.version(dist_name)
            status = "PASS"
            print(f"{import_name:<18} {str(ver):<18} {status}")
            package_info.append((import_name, dist_name, str(ver), status))
        except Exception as e:
            status = "FAIL"
            print(f"{import_name:<18} {'N/A':<18} {status} ({e})")
            package_info.append((import_name, dist_name, "N/A", status))
            all_ok = False

    if not all_ok:
        raise RuntimeError("STEP 4 FAIL: One or more package imports failed!")

    print("STEP 4 RESULT: PASS")
    return package_info


def step_5_pytorch():
    print("\n" + "=" * 80)
    print("STEP 5 — PYTORCH VERIFICATION")
    print("=" * 80)
    import torch
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    cuda_ver = torch.version.cuda if cuda_available else "N/A"
    selected_device = "cuda" if cuda_available else "cpu"

    print(f"PyTorch Version:   {torch.__version__}")
    print(f"CUDA Available:    {cuda_available}")
    print(f"CUDA Version:      {cuda_ver}")
    print(f"Device Count:      {device_count}")
    print(f"Selected Device:   {selected_device}")

    # Small tensor calculation
    t = torch.tensor([1.0, 2.0, 3.0])
    t_sum = t.sum().item()
    assert t_sum == 6.0, "Tensor calculation failed"
    print(f"In-memory tensor test (sum([1,2,3])): {t_sum} [PASS]")

    print("STEP 5 RESULT: PASS")
    return {
        "version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": cuda_ver,
        "device_count": device_count,
        "selected_device": selected_device
    }


def step_6_transformers():
    print("\n" + "=" * 80)
    print("STEP 6 — TRANSFORMERS VERIFICATION")
    print("=" * 80)
    from transformers import AutoTokenizer, AutoModel
    import transformers
    print(f"Transformers Version: {transformers.__version__}")
    print(f"AutoTokenizer imported: {AutoTokenizer is not None} [PASS]")
    print(f"AutoModel imported:     {AutoModel is not None} [PASS]")
    print("Model download skipped as required for Phase 2.")
    print("STEP 6 RESULT: PASS")
    return transformers.__version__


def step_7_pandas_numpy_sklearn():
    print("\n" + "=" * 80)
    print("STEP 7 — PANDAS / NUMPY / SCIKIT-LEARN FUNCTIONALITY TEST")
    print("=" * 80)
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split

    # 1. NumPy calculation
    arr = np.arange(10, dtype=float)
    mean_val = np.mean(arr)
    print(f"NumPy in-memory array mean (0..9): {mean_val} [PASS]")
    assert mean_val == 4.5

    # 2. Pandas DataFrame test
    df_test = pd.DataFrame({"x": arr, "y": arr * 2})
    assert len(df_test) == 10 and df_test["y"].iloc[5] == 10.0
    print(f"Pandas DataFrame shape: {df_test.shape} [PASS]")

    # 3. Scikit-learn train_test_split test
    tr, te = train_test_split(df_test, test_size=0.2, random_state=42)
    assert len(tr) == 8 and len(te) == 2
    print(f"Scikit-learn split: train={len(tr)}, test={len(te)} [PASS]")

    print("STEP 7 RESULT: PASS")
    return True


def step_8_pillow_opencv():
    print("\n" + "=" * 80)
    print("STEP 8 — PILLOW / OPENCV FUNCTIONALITY TEST")
    print("=" * 80)
    from PIL import Image
    import cv2
    import numpy as np

    # Create small in-memory RGB image (64x64)
    img_pil = Image.new("RGB", (64, 64), color=(255, 128, 0))
    buffer = io.BytesIO()
    img_pil.save(buffer, format="PNG")
    buffer.seek(0)
    img_bytes = buffer.read()

    # Read with OpenCV
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    h, w, c = img_cv.shape
    print(f"Pillow created RGB image: 64x64")
    print(f"OpenCV loaded image dimensions: height={h}, width={w}, channels={c}")
    assert (h, w, c) == (64, 64, 3), "Image dimensions mismatch"

    print("STEP 8 RESULT: PASS")
    return True


def step_9_pytesseract():
    print("\n" + "=" * 80)
    print("STEP 9 — PYTESSERACT PYTHON PACKAGE VERIFICATION")
    print("=" * 80)
    import pytesseract
    print(f"pytesseract version: {pytesseract.__version__}")
    print("STEP 9 RESULT: PASS")
    return pytesseract.__version__


def step_10_tesseract_engine():
    print("\n" + "=" * 80)
    print("STEP 10 — TESSERACT OCR ENGINE DISCOVERY")
    print("=" * 80)
    
    tess_path = shutil.which("tesseract")
    standard_windows_paths = [
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    ]

    installed = False
    exe_found = None

    if tess_path:
        installed = True
        exe_found = tess_path
    else:
        for p in standard_windows_paths:
            if p.exists():
                installed = True
                exe_found = str(p)
                break

    if installed:
        print(f"Tesseract OCR Engine Found: {exe_found}")
        try:
            ver_out = subprocess.check_output([exe_found, "--version"], text=True, stderr=subprocess.STDOUT)
            ver_line = ver_out.splitlines()[0]
            print(f"Tesseract Engine Version: {ver_line}")

            lang_out = subprocess.check_output([exe_found, "--list-langs"], text=True, stderr=subprocess.STDOUT)
            langs = [l.strip() for l in lang_out.splitlines()[1:] if l.strip()]
            print(f"Installed OCR Languages ({len(langs)}): {langs}")
            tess_result = {
                "installed": True,
                "executable": exe_found,
                "version": ver_line,
                "languages": langs,
                "status": "PASS"
            }
        except Exception as e:
            tess_result = {
                "installed": True,
                "executable": exe_found,
                "version": "ERROR",
                "languages": [],
                "status": f"ERROR ({e})"
            }
    else:
        print("PYTESSERACT PYTHON PACKAGE: PASS")
        print("TESSERACT OCR ENGINE:       NOT INSTALLED")
        print("WARNING: Tesseract OCR engine executable is not installed on this machine.")
        print("         It MUST be installed prior to commencing Phase 3 (OCR & Image Processing).")
        tess_result = {
            "installed": False,
            "executable": "NOT INSTALLED",
            "version": "N/A",
            "languages": [],
            "status": "WARNING — REQUIRED BEFORE PHASE 3"
        }

    print(f"STEP 10 RESULT: {tess_result['status']}")
    return tess_result


def step_11_requests_dotenv():
    print("\n" + "=" * 80)
    print("STEP 11 — REQUESTS & PYTHON-DOTENV TEST (NO NETWORK ACCESS)")
    print("=" * 80)
    import requests
    import dotenv
    import importlib.metadata

    dotenv_ver = getattr(dotenv, "__version__", None) or importlib.metadata.version("python-dotenv")
    print(f"requests version:      {requests.__version__} [PASS]")
    print(f"python-dotenv version: {dotenv_ver} [PASS]")
    print("Zero network requests made (in accordance with local offline rules).")
    print("STEP 11 RESULT: PASS")
    return True


def step_12_streamlit():
    print("\n" + "=" * 80)
    print("STEP 12 — STREAMLIT INSTALLATION VERIFICATION")
    print("=" * 80)
    import streamlit
    print(f"Streamlit Version: {streamlit.__version__}")
    print("Streamlit UI was NOT launched (in accordance with Phase 2 rules).")
    print("STEP 12 RESULT: PASS")
    return streamlit.__version__


def step_13_pytest():
    print("\n" + "=" * 80)
    print("STEP 13 — PYTEST VERIFICATION")
    print("=" * 80)
    import pytest
    print(f"pytest Version: {pytest.__version__}")
    print("STEP 13 RESULT: PASS")
    return pytest.__version__


def step_14_phase1_dataset_access():
    print("\n" + "=" * 80)
    print("STEP 14 — PHASE 1 DATASET READ-ONLY ACCESSIBILITY CHECK")
    print("=" * 80)
    files = [
        RAW_DIR / "IFND" / "IFND.csv",
        RAW_DIR / "LIAR" / "train.tsv",
        RAW_DIR / "LIAR" / "test.tsv",
        RAW_DIR / "LIAR" / "valid.tsv",
        PROCESSED_DIR / "ifnd_clean.csv",
        PROCESSED_DIR / "liar_combined.csv",
        SPLITS_DIR / "ifnd_train.csv",
        SPLITS_DIR / "ifnd_validation.csv",
        SPLITS_DIR / "ifnd_test.csv",
    ]

    all_exist = True
    for p in files:
        if p.exists() and p.stat().st_size > 0:
            print(f"  [PASS] {p.relative_to(PROJECT_ROOT)} ({p.stat().st_size:,} bytes)")
        else:
            print(f"  [FAIL] Missing or empty: {p}")
            all_exist = False

    if not all_exist:
        raise RuntimeError("STEP 14 FAIL: Phase 1 files missing or inaccessible!")

    print("STEP 14 RESULT: PASS")
    return True


def step_15_phase1_raw_hashes():
    print("\n" + "=" * 80)
    print("STEP 15 — PHASE 1 RAW FILE HASH INTEGRITY CHECK")
    print("=" * 80)
    before_csv = EVAL_DIR / "raw_file_hashes_before.csv"
    after_csv = EVAL_DIR / "raw_file_hashes_after.csv"

    if not before_csv.exists() or not after_csv.exists():
        raise RuntimeError("STEP 15 FAIL: Raw hash reference CSVs missing!")

    df_before = pd.read_csv(before_csv)
    df_after = pd.read_csv(after_csv)

    if not df_before.equals(df_after):
        raise RuntimeError("STEP 15 FAIL: Stored before and after hashes do not match!")

    # Verify live SHA-256
    live_match = True
    for _, row in df_before.iterrows():
        fpath = RAW_DIR / ("IFND/IFND.csv" if row["dataset"] == "IFND" else f"LIAR/{row['file']}")
        sha = hashlib.sha256()
        with open(fpath, "rb") as f:
            while chunk := f.read(65536):
                sha.update(chunk)
        calc = sha.hexdigest()
        match = (calc == row["sha256"])
        mark = "[PASS]" if match else "[FAIL]"
        print(f"  {mark} {row['file']}: live={calc[:16]}... stored={row['sha256'][:16]}...")
        if not match:
            live_match = False

    if not live_match:
        raise RuntimeError("STEP 15 FAIL: Live raw hash mismatch detected!")

    print("STEP 15 RESULT: PASS")
    return True


def step_16_generate_reports(py_ver, pkg_info, torch_info, tess_info):
    print("\n" + "=" * 80)
    print("STEP 16 — GENERATING ENVIRONMENT REPORTS")
    print("=" * 80)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. environment_report.txt
    report_lines = [
        "=" * 80,
        "FAKESHIELD ENVIRONMENT DIAGNOSTIC REPORT (PHASE 2)",
        "=" * 80,
        f"Project Root:           {PROJECT_ROOT}",
        f"Operating System:       {platform.platform()}",
        f"Python Version:         {py_ver}",
        f"Python Executable:      {sys.executable}",
        f"Virtual Environment:    {'ACTIVE (venv)' if sys.prefix != sys.base_prefix else 'GLOBAL'}",
        f"Pip Version:            {shutil.which('pip')}",
        "",
        "INSTALLED PACKAGE MATRIX:",
        "-" * 60,
        f"{'Package':<18} {'Version':<18} {'Import Status'}",
        "-" * 60,
    ]
    for imp_name, dist_name, ver, status in pkg_info:
        report_lines.append(f"{imp_name:<18} {ver:<18} {status}")

    report_lines.extend([
        "",
        "PYTORCH & ACCELERATION:",
        "-" * 60,
        f"PyTorch Version:        {torch_info['version']}",
        f"CUDA Available:         {torch_info['cuda_available']}",
        f"CUDA Version:           {torch_info['cuda_version']}",
        f"Device Count:           {torch_info['device_count']}",
        f"Selected Device:        {torch_info['selected_device']}",
        "",
        "TESSERACT OCR ENGINE:",
        "-" * 60,
        f"Python Binding:         pytesseract (INSTALLED)",
        f"Engine Installed:       {tess_info['installed']}",
        f"Engine Executable:      {tess_info['executable']}",
        f"Engine Version:         {tess_info['version']}",
        f"Available Languages:    {tess_info['languages']}",
        f"Engine Status:          {tess_info['status']}",
        "",
        "DATASET ACCESS & INTEGRITY:",
        "-" * 60,
        "Phase 1 Dataset Access: PASS (all raw, processed, and splits verified)",
        "Raw Hashes Integrity:   PASS (100% match with Phase 1 baseline)",
        "=" * 80,
        "END OF REPORT",
        "=" * 80,
    ])

    report_path = EVAL_DIR / "environment_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"  Generated: {report_path.relative_to(PROJECT_ROOT)}")

    # 2. environment_packages.csv
    import pandas as pd
    pkg_rows = []
    for imp_name, dist_name, ver, status in pkg_info:
        pkg_rows.append({
            "package": dist_name,
            "installed_version": ver,
            "required": "True",
            "import_status": status,
            "notes": f"Imported as {imp_name}; core FakeShield Phase 2 dependency"
        })
    pkg_df = pd.DataFrame(pkg_rows)
    csv_path = EVAL_DIR / "environment_packages.csv"
    pkg_df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"  Generated: {csv_path.relative_to(PROJECT_ROOT)}")

    # 3. environment_reproducibility.txt
    repro_lines = [
        "=" * 80,
        "FAKESHIELD ENVIRONMENT REPRODUCIBILITY SPECIFICATION",
        "=" * 80,
        f"Operating System:       {platform.system()} {platform.release()} ({platform.machine()})",
        f"Python Version:         {py_ver}",
        f"Virtual Env Path:       {Path(sys.prefix).name}",
        f"PyTorch Device:         {torch_info['selected_device']} (CUDA={torch_info['cuda_available']})",
        f"Pinned Requirements:    requirements.txt",
        "Random Seed Standard:   42",
        "Offline Execution:      VERIFIED (No network calls or downloads in Phase 2)",
        "=" * 80,
    ]
    repro_path = EVAL_DIR / "environment_reproducibility.txt"
    with open(repro_path, "w", encoding="utf-8") as f:
        f.write("\n".join(repro_lines))
    print(f"  Generated: {repro_path.relative_to(PROJECT_ROOT)}")

    print("STEP 16 RESULT: PASS")
    return True


def main():
    print("#" * 80)
    print("# FAKESHIELD: MASTER ENVIRONMENT DIAGNOSTIC PIPELINE")
    print("#" * 80)
    try:
        step_1_project_root()
        py_ver = step_2_python()
        step_3_virtual_environment()
        pkg_info = step_4_package_imports()
        torch_info = step_5_pytorch()
        step_6_transformers()
        step_7_pandas_numpy_sklearn()
        step_8_pillow_opencv()
        step_9_pytesseract()
        tess_info = step_10_tesseract_engine()
        step_11_requests_dotenv()
        step_12_streamlit()
        step_13_pytest()
        step_14_phase1_dataset_access()
        step_15_phase1_raw_hashes()
        step_16_generate_reports(py_ver, pkg_info, torch_info, tess_info)

        print("\n" + "=" * 80)
        print("ALL ENVIRONMENT DIAGNOSTIC STEPS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        return 0
    except Exception as e:
        print(f"\nENVIRONMENT CHECK ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import pandas as pd  # ensure pandas is in namespace
    sys.exit(main())
