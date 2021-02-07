#!/usr/bin/env python
from setuptools import find_packages, setup

"""The setup script."""
try:  # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError:  # for pip <= 9.0.3
    from pip.req import parse_requirements


with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()


## workaround derived from: https://github.com/pypa/pip/issues/7645#issuecomment-578210649
parsed_requirements = parse_requirements("requirements/prod.txt", session="workaround")

parsed_test_requirements = parse_requirements(
    "requirements/test.txt", session="workaround"
)


requirements = [str(ir.requirement) for ir in parsed_requirements]
test_requirements = [str(tr.requirement) for tr in parsed_test_requirements]

setup_requirements = []


setup(
    author="Alexandros Dorodoulis, Amalia Matei, Jesper Lund Petersen",
    author_email="",
    python_requires=">=3.5",
    classifiers=[
        # As from http://pypi.python.org/pypi?%3Aaction=list_classifiers
        # 'Development Status :: 1 - Planning',
        "Development Status :: 2 - Pre-Alpha",
        # 'Development Status :: 3 - Alpha',
        # "Development Status :: 4 - Beta",
        # 'Development Status :: 5 - Production/Stable',
        # 'Development Status :: 6 - Mature',
        # 'Development Status :: 7 - Inactive',
        "Intended Audience :: Developers",
        "Environment :: Console",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Main model for master thesis",
    entry_points={
        "console_scripts": [
            "the_wizard_express=the_wizard_express.cli:main",
        ],
    },
    install_requires=requirements,
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="the_wizard_express",
    name="the_wizard_express",
    packages=find_packages(include=["the_wizard_express", "the_wizard_express.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/alexdor/the_wizard_express",
    version="0.0.1",
    zip_safe=False,
)
