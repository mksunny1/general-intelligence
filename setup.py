from setuptools import setup, find_packages

setup(
    name="general-intelligence",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["numpy"],
    author="Oluwaseyi Shoboyejo",
    description="Self-organizing knowledge systems for structural pattern learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/general-intelligence",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)