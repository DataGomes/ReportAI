import sys
import subprocess
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop

class CustomInstallCommand(install):
    def run(self):
        subprocess.check_call([sys.executable, 'install_dependencies.py'])
        install.run(self)

class CustomDevelopCommand(develop):
    def run(self):
        subprocess.check_call([sys.executable, 'install_dependencies.py'])
        develop.run(self)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name='report_generator',
    version='0.1',
    author='Andre Gomes',
    author_email='andmagal@gmail.com',
    description='A library for generating scientific reports based on literature queries',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/report_generator",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'generate_report=report_generator.cli:main',
        ],
    },
    include_package_data=True,
    cmdclass={
        'install': CustomInstallCommand,
        'develop': CustomDevelopCommand,
    },
)