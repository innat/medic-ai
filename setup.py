import re

from setuptools import find_packages, setup


def get_install_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as f:
        reqs = [x.strip() for x in f.read().splitlines()]
    reqs = [x for x in reqs if not x.startswith("#")]
    return reqs


def get_medicai_version():
    with open("medicai/__init__.py", "r") as f:
        version = re.search(
            r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(), re.MULTILINE
        ).group(1)
    return version


def get_long_description():
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
    return long_description


setup(
    name="medicai",
    version=get_medicai_version(),
    author="innat",
    author_email="innat.dev@gmail.com",
    url="https://github.com/innat/medic-ai",
    package_dir={"": "."},
    packages=find_packages(
        where=".",
        include=["medicai", "medicai.*"],
        exclude=("test", "dataloader", "notebooks"),
    ),
    extras_require={
        "tensorflow": ["tensorflow[and-cuda]"],
        "jax": ["jax[cuda12_local]"],
        "torch": ["torch==2.6.0+cu124"],
        "all": [
            "tensorflow[and-cuda]",
            "jax[cuda12_local]",
            "torch==2.6.0+cu124",
        ],
    },
    python_requires=">=3.6",
    install_requires=get_install_requirements(),
    setup_requires=["wheel"],  # avoid building error when pip is not updated
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: Apache Software License",
    ],
    entry_points={"console_scripts": ["medicai = src.medicai.cli:cli"]},
)
