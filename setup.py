from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Filter out pip-specific entries like -e .
requirements = [req for req in requirements if not req.startswith('-e') and req.strip()]

setup(
    name="Customer_Feedback_Analyzer",
    version="0.1",
    author="beniaminenahid",
    packages=find_packages(),
    install_requires=requirements,
)