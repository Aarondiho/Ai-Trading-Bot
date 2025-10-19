"""
Platform Mind Reader - Installation Setup
"""

from setuptools import setup, find_packages

setup(
    name="platform-mind-reader",
    version="1.0.0",
    description="AI Trading System that thinks like Deriv's platform algorithms",
    author="Platform Mind Reader Team",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0", 
        "scikit-learn>=1.0.0",
        "tensorflow>=2.11.0",
        "talib-binary>=0.4.28",
        "websockets>=10.0",
        "requests>=2.28.0",
        "scipy>=1.7.0",
        "statsmodels>=0.13.0",
        "ruptures>=1.1.0",
        "markovify>=0.9.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
