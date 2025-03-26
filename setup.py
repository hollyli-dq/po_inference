from setuptools import setup, find_packages

setup(
    name="po_inference",
    version="0.1.0",
    author="Dongqing Li",
    author_email="kell7733@ox.ac.uk",
    description="A Markov Chain Monte Carlo approach for inferring partial orders from data",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "matplotlib",
        "pyyaml",
        "networkx",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
        "jupyter>=1.0.0",
    ],
    extras_require={
        'dev': [
            "pytest>=6.2.0",
        ],
    },
    entry_points={
        'console_scripts': [
            'po-inference=src.cli:main',
        ],
    },
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords="bayesian, partial order, mcmc, inference",
    url="https://github.com/yourusername/po_inference",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
) 