from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="crop-yield-prediction",
    version="1.0.0",
    author="Muhammad Ahsan Tariq",
    author_email="tariqma@ulster.ac.uk",
    description="ML-powered agricultural yield prediction system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MAT-Alpha-375/crop-yield-prediction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "crop-yield-prediction=app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.csv", "*.pkl", "*.cbm"],
    },
)
