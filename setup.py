from setuptools import setup, find_packages

setup(
    name='pillar_project',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "seaborn",
        "matplotlib",
    ],
    entry_points={
        'console_scripts': [
            # Add command line scripts here
        ],
    },
    author='Daniel Zeiberg',
    author_email='d.zeiberg@northeastern.edu',
    description='IGVF Coding Variant Focus Group Pillar Project',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Dzeiberg/pillar_project',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)