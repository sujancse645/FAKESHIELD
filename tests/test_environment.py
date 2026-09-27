"""
test_environment.py - FakeShield Automated Environment Unit Tests

Tests verify:
1. Python version >= 3.10
2. pandas import
3. numpy import
4. sklearn import
5. Pillow import
6. OpenCV import
7. pytesseract import
8. transformers import
9. torch import
10. requests import
11. dotenv import
12. streamlit import
13. pytest environment works
"""

import sys
import pytest


def test_python_version():
    """Verify Python version is at least 3.10."""
    major, minor = sys.version_info[:2]
    assert (major == 3 and minor >= 10), f"Python version must be >= 3.10, got {sys.version}"


def test_pandas_import():
    """Verify pandas can be imported."""
    import pandas as pd
    assert pd.__version__ is not None


def test_numpy_import():
    """Verify numpy can be imported."""
    import numpy as np
    assert np.__version__ is not None


def test_sklearn_import():
    """Verify scikit-learn can be imported."""
    import sklearn
    assert sklearn.__version__ is not None


def test_pillow_import():
    """Verify Pillow can be imported."""
    import PIL
    from PIL import Image
    assert PIL.__version__ is not None


def test_opencv_import():
    """Verify OpenCV can be imported."""
    import cv2
    assert cv2.__version__ is not None


def test_pytesseract_import():
    """Verify pytesseract can be imported."""
    import pytesseract
    assert pytesseract.__version__ is not None


def test_transformers_import():
    """Verify transformers can be imported without model downloads."""
    import transformers
    from transformers import AutoTokenizer, AutoModel
    assert transformers.__version__ is not None
    assert AutoTokenizer is not None
    assert AutoModel is not None


def test_torch_import():
    """Verify torch can be imported."""
    import torch
    assert torch.__version__ is not None


def test_requests_import():
    """Verify requests can be imported without making network calls."""
    import requests
    assert requests.__version__ is not None


def test_dotenv_import():
    """Verify python-dotenv can be imported."""
    import dotenv
    import importlib.metadata
    assert dotenv.load_dotenv is not None
    ver = importlib.metadata.version("python-dotenv")
    assert ver is not None


def test_streamlit_import():
    """Verify streamlit can be imported."""
    import streamlit
    assert streamlit.__version__ is not None


def test_pytest_environment_works():
    """Verify that pytest execution environment functions properly."""
    assert True
