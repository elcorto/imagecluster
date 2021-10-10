import os
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.rst"), encoding="utf-8") as fd:
    long_description = fd.read()


setup(
    name="imagecluster",
    version="0.4.1",
    description="cluster images based on image content using a pre-trained "
    "deep neural network and hierarchical clustering",
    long_description=long_description,
    url="https://github.com/elcorto/imagecluster",
    author="Steve Schmerler",
    author_email="git@elcorto.com",
    license="BSD 3-Clause",
    keywords="image cluster vgg16 deep-learning",
    packages=["imagecluster"],
    install_requires=open("requirements.txt").read().splitlines(),
)
