import os
import pkg_resources
from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

setup(
    name='multimodal',
    version='0.1.0',
    description='SOTA Multimodal Models for various tasks',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/sithu31296/multimodal',
    author='sithu3',
    author_email='sithu31296@gmail.com',
    packages=find_packages(exclude=['tests*']),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    keywords=['python'],
    include_package_data=True
)