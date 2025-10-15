"""Tests for model trainer"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def test_model_trainer_import():
    """Test that we can import the model trainer."""
    from models.model_trainer import ModelTrainer
    assert ModelTrainer is not None

def test_data_preparation():
    """Test data preparation."""
    from models.model_trainer import ModelTrainer

    # Create sample data
    data = pd.DataFrame({
        'Country': ['USA', 'Canada'],
        'Year': [2015, 2015],
        'Status': ['Developed', 'Developed'],
        'Life expectancy ': [78.5, 82.2],
        'GDP': [55000, 45000],
        'Schooling': [13, 14]
    })

    trainer = ModelTrainer()
    X, y = trainer.prepare_data(data)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y)
