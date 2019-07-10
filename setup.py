# Always prefer setuptools over distutils
from setuptools import setup

# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="tfpyth",
    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version="1.0.1",
    description="Putting TensorFlow back in PyTorch, back in Tensorflow (differentiable TensorFlow PyTorch adapters).",
    # Fix windows newlines.
    long_description=long_description.replace("\r\n", "\n"),
    long_description_content_type="text/markdown",
    # The project's main homepage.
    url="https://github.com/blackhc/tfpyth",
    # Author details
    author="Andreas @blackhc Kirsch",
    author_email="blackhc+tfpyth@gmail.com",
    # Choose your license
    license="MIT",
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 4 - Beta",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Libraries :: Python Modules",
        # Pick your license as you wish (should match "license" above)
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
    # What does your project relate to?
    keywords="ml machine learning",
    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=["tfpyth"],
    #package_dir={"": ""},
    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=["tensorflow-gpu~=1.14", "torch~=1.1"],
    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={"dev": ["check-manifest"], "test": ["coverage", "pytest"]},
    setup_requires=["pytest-runner"],
)
