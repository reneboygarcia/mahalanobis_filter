from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="mahalanobis_filter",
    version="0.1.0",
    author="reneboygarcia",
    author_email="reneboygarcia@gmail.com",
    description="Mahalanobis distance filter for quality control",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/reneboygarcia/mahalanobis_filter",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "mahalanobis-dashboard=mahalanobis_filter.mahalanobis_filter_dash:run_server",
        ],
    },
)
