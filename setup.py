""" setup.py

Tells Python how to build and install package. 

"""

from setuptools import setup, find_packages

# Read the requirements from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="spectrum_sharing", 
    version="0.0.1",       
    description="Using NVIDIA Sionna to train DRL agents for spectrum sharing.",
    author="Oscar Dilley",
    author_email="oscar.dilley@bristol.ac.uk",
    url="https://github.com/oscardilley/spectrum_sharing", 
    packages=find_packages(),  # Automatically find all packages in the repo
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    license="Apache License 2.0",
)
