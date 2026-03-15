from setuptools import setup, find_packages

setup(
    name="atlas-nas",
    version="0.1.0",
    description="ATLAS: Adaptive Three-phase Landscape-Aware Search for NAS",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Anonymous",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.7",
        "scikit-learn>=1.0",
    ],
    extras_require={
        "dev": ["pytest", "matplotlib"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
