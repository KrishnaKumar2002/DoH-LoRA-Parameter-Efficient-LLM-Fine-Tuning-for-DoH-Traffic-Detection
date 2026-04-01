"""
Setup configuration for DoH-LoRA package.
"""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="doh-lora",
    version="1.0.0",
    author="Krishna Kumar M",
    author_email="cmkkcse@gmail.com",
    description="Parameter-Efficient LLM Fine-Tuning for DNS-over-HTTPS Traffic Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KrishnaKumar2002/DoH-LoRA-Parameter-Efficient-LLM-Fine-Tuning-for-DoH-Traffic-Detection",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "peft>=0.10.0",
        "accelerate>=0.30.0",
        "datasets>=2.16.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.5.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.13.0",
        "scipy>=1.11.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": [
            "black>=24.0.0",
            "isort>=5.13.0",
            "flake8>=7.0.0",
            "mypy>=1.8.0",
            "pytest>=7.4.4",
        ],
    },
    entry_points={
        "console_scripts": [
            "doh-lora=doh_lora.main:main",
        ],
    },
)
