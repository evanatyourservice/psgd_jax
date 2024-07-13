from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

if __name__ == "__main__":
    setup(
        name="psgd_jax",
        version="0.0.1",
        author="Evan Walters, Omead Pooladzandi, Xi-Lin Li",
        description="PSGD optimizer in JAX",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/evanatyourservice/psgd_jax",
        packages=find_packages(),
        license="Creative Commons Attribution-ShareAlike 4.0 International License",
        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Scientific/Engineering :: Mathematics",
        ],
        install_requires=required,
        python_requires=">=3.10",
    )
