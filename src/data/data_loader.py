"""Advanced Data Loader Module"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Dict, Any, Union
import requests

logger = logging.getLogger(__name__)

class DataLoader:
    """Advanced data loader with validation and preprocessing."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def load_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load and validate dataset."""
        file_path = Path(file_path)

        if not file_path.exists():
            logger.warning(f"Data file not found: {file_path}")
            logger.info("Creating sample dataset for demonstration...")
            return self._create_sample_data()

        logger.info(f"Loading data from: {file_path}")

        try:
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")

            logger.info(f"✅ Data loaded successfully. Shape: {df.shape}")
            self._validate_data(df)
            return df

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return self._create_sample_data()

    def _validate_data(self, df: pd.DataFrame) -> None:
        """Validate dataset structure and content."""
        required_columns = ['Life expectancy ', 'Country', 'Year']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        if df.empty:
            raise ValueError("Dataset is empty")

        # Log basic statistics
        logger.info(f"Dataset validation passed:")
        logger.info(f"  - Shape: {df.shape}")
        logger.info(f"  - Countries: {df['Country'].nunique()}")
        logger.info(f"  - Years: {df['Year'].min()}-{df['Year'].max()}")
        logger.info(f"  - Life expectancy range: {df['Life expectancy '].min():.1f}-{df['Life expectancy '].max():.1f}")

    def _create_sample_data(self) -> pd.DataFrame:
        """Create comprehensive sample dataset."""
        np.random.seed(42)

        countries = [
            'United States', 'Germany', 'Japan', 'Brazil', 'India', 
            'Nigeria', 'Kenya', 'France', 'Canada', 'Australia',
            'China', 'Russia', 'South Africa', 'Mexico', 'Argentina'
        ]

        years = list(range(2000, 2016))
        data = []

        for country in countries:
            is_developed = country in ['United States', 'Germany', 'Japan', 'France', 'Canada', 'Australia']
            base_life_exp = 75 if is_developed else 62

            for year in years:
                # Create realistic progression over time
                time_trend = (year - 2000) * 0.25
                noise = np.random.normal(0, 2)
                life_exp = base_life_exp + time_trend + noise
                life_exp = max(45, min(90, life_exp))  # Realistic bounds

                record = {
                    'Country': country,
                    'Year': year,
                    'Status': 'Developed' if is_developed else 'Developing',
                    'Life expectancy ': life_exp,
                    'Adult Mortality': np.random.normal(120 if is_developed else 200, 50),
                    'infant deaths': max(0, int(np.random.poisson(15 if is_developed else 35))),
                    'Alcohol': max(0, np.random.exponential(3 if is_developed else 1.5)),
                    'percentage expenditure': max(0, np.random.exponential(200 if is_developed else 50)),
                    'Hepatitis B': np.random.normal(85 if is_developed else 70, 15),
                    'Measles ': max(0, int(np.random.poisson(100 if not is_developed else 20))),
                    ' BMI ': np.random.normal(26 if is_developed else 23, 4),
                    'under-five deaths ': max(0, int(np.random.poisson(18 if is_developed else 45))),
                    'Polio': np.random.normal(88 if is_developed else 75, 12),
                    'Total expenditure': np.random.normal(8 if is_developed else 4, 2),
                    'Diphtheria ': np.random.normal(87 if is_developed else 73, 13),
                    ' HIV/AIDS': max(0, np.random.exponential(0.5 if is_developed else 2)),
                    'GDP': max(100, np.random.exponential(15000 if is_developed else 2000)),
                    'Population': max(100000, int(np.random.exponential(25000000))),
                    ' thinness  1-19 years': max(0, np.random.normal(3 if is_developed else 8, 3)),
                    ' thinness 5-9 years': max(0, np.random.normal(3 if is_developed else 8, 3)),
                    'Income composition of resources': np.clip(np.random.beta(3 if is_developed else 2, 2), 0, 1),
                    'Schooling': np.random.normal(13 if is_developed else 9, 2)
                }
                data.append(record)

        df = pd.DataFrame(data)
        logger.info(f"✅ Sample dataset created with {len(df)} records")
        return df
