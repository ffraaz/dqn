from pkg_resources import parse_requirements
from setuptools import setup

with open("requirements.txt") as f:
    requirements = [str(req) for req in parse_requirements(f)]

setup(
    name="dqn",
    version="0.1.0",
    py_modules=["dqn"],
    python_requires=">=3.9",
    install_requires=requirements,
)
