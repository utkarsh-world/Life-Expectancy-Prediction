#!/usr/bin/env python3
"""
Setup script for Life Expectancy Prediction Enhanced
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
README_PATH = Path(__file__).parent / "README.md"
if README_PATH.exists():
    with open(README_PATH, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "Enhanced machine learning project for predicting global life expectancy"

# Read requirements
REQUIREMENTS_PATH = Path(__file__).parent / "requirements.txt"
if REQUIREMENTS_PATH.exists():
    with open(REQUIREMENTS_PATH, "r", encoding="utf-8") as fh:
        requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
else:
    requirements = [
        "pandas>=1.5.0",
        "scikit-learn>=1.1.0", 
        "streamlit>=1.12.0",
        "flask>=2.2.0"
    ]

setup(
    name="life-expectancy-prediction-enhanced",
    version="2.0.0",
    author="Utkarsh Sharma",
    author_email="utkarsh@dataanalysis.com",
    description="Enhanced machine learning project for predicting global life expectancy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/utkarsh-world/Life-Expectancy-Prediction-Enhanced",
    project_urls={
        "Bug Tracker": "https://github.com/utkarsh-world/Life-Expectancy-Prediction-Enhanced/issues",
        "Documentation": "https://github.com/utkarsh-world/Life-Expectancy-Prediction-Enhanced/docs",
        "Source Code": "https://github.com/utkarsh-world/Life-Expectancy-Prediction-Enhanced",
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.1.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "isort>=5.10.0",
            "mypy>=0.991",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "mkdocs>=1.4.0",
            "mkdocs-material>=8.5.0",
        ],
        "deployment": [
            "gunicorn>=20.1.0",
            "docker>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "life-expectancy-predict=main:main",
            "life-expectancy-api=src.api.app:main",
            "life-expectancy-dashboard=src.app.streamlit_app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["configs/*.yaml", "data/*.csv", "models/*.joblib"],
    },
    zip_safe=False,
    keywords=[
        "machine-learning", "life-expectancy", "health-analytics", 
        "data-science", "prediction", "WHO", "healthcare", "streamlit", 
        "docker", "api", "random-forest", "deep-learning"
    ],
)
