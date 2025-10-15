"""Tests for data loader"""
import pytest
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def test_data_loader_import():
    """Test that we can import the data loader."""
    from data.data_loader import DataLoader
    assert DataLoader is not None

def test_sample_data_creation():
    """Test sample data creation."""
    from data.data_loader import DataLoader

    loader = DataLoader()
    df = loader._create_sample_data()

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert 'Life expectancy ' in df.columns
    assert 'Country' in df.columns
