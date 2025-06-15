"""
Setup script for Stability Analysis of Gradient Descent in Super-Resolution
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="stability-gradient-descent-sr",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Stability analysis of gradient descent in super-resolution with AdaFM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/Stability_GradientDescent_SuperResolution",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "isort>=5.0.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
            "sphinx-autodoc-typehints>=1.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "run-stability-analysis=experiments.main_stability_analysis:main",
            "run-theoretical-validation=experiments.theoretical_validation:main",
            "run-minimax-demo=experiments.minimax_demo:main",
        ],
    },
)