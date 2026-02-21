"""Package configuration for violingen."""

from setuptools import setup, find_packages

setup(
    name="violingen",
    version="0.1.0",
    description="A tiny autogenerative violin model",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
    ],
)
