from setuptools import setup, find_packages


# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


setup(
    name="YASE",
    version="1.0.1",
    packages=find_packages(),
    license='MIT',
    author='Matt Ranger',
    url='https://github.com/VHRanger/YASE/',
    long_description=long_description,
    long_description_content_type='text/markdown',
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        '': ['README.md', 'requirements.txt']
    },

    install_requires=[
        # TODO: match to requirements.txt
        'gensim',
        'numpy',
        'pandas',
        'scikit-learn',
		'pytest',
		'pytest-cov'
      ],
)
