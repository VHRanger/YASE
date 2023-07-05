from setuptools import setup, find_packages


setup(
    name="YASE",
    version="1.0.0",
    packages=find_packages(),
    description="Yet Another Sentence Embedding Library",

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
