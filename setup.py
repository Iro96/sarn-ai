from setuptools import setup, find_packages

setup(
    name="sarn",
    version="0.1.1",
    author="Iro Sowara",
    author_email= "bruh8080p@gmail.com",
    description="SARN - Self-Adaptive Rewiring Neural Network",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Iro96/SarnAI",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy",
    ],
)
