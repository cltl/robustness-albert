from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f.readlines()]


setup(
    name="robustness_albert",
    version="1.0",
    author="Urja Khurana",
    author_email="u.khurana@vu.nl",
    url="",
    description="Code for paper testing robustness of ALBERT (with SWA) on a sentiment analysis task using CheckList.",
    packages=find_packages(),
    install_requires=requirements,
)
