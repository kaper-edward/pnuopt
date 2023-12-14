from setuptools import setup, find_packages

setup(
    name='pnuopt',
    version='0.1.0',
    author='Edward Kim',
    author_email='tykim@pusan.ac.kr',
    packages=find_packages(),
    url='https://github.com/kaper-edward/pnuopt.git',
    license='LICENSE.txt',
    description='An optimization package',
    long_description=open('README.md').read(),
    python_requires='>3.10',
    install_requires=[
        "numpy >= 1.20",
        "scikit-learn >= 1.2.0",
    ],
)
