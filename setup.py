"""
Setup script for Stability Analysis project
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
    author_email="your.email@domain.com",
    description="Stability Analysis for Gradient Descent in Super-Resolution",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Stability_GradientDescent_SuperResolution",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "isort>=5.10.0",
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
        ],
        "viz": [
            "plotly>=5.0.0",
            "tensorboard>=2.10.0",
        ],
        "full": [
            "scikit-image>=0.19.0",
            "opencv-python>=4.6.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "stability-experiment=experiments.experiments_main:main",
            "stability-theoretical=experiments.experiments_theoretical:main",
            "stability-minimax=experiments.experiments_minimax_demo:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt", "*.md"],
    },
    zip_safe=False,
)
