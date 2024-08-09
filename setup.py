from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="occupancy_predictor",
    version="0.1.0",
    author="Javad Yazdanfar",
    author_email="yazdanfar.de@gmail.com",
    description="A package to predict room occupancy based on environmental sensors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yazdanfar/Occupancy-Predictor",
    packages=find_packages(exclude=["tests", "scripts"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "joblib",
        "psutil",
    ],
    extras_require={
        'test': ['pytest'],
    },
)