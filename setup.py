from io import open
from setuptools import find_packages, setup

setup(
    name="da_bert_score",
    version='0.0.1',
    author="Runzhe Zhan",
    author_email="nlp2ct.runzhe@gmail.com",
    description="Difficulty-Aware Machine Translation Evaluation implementation, ACL2021 (based on BERTScore)",
    long_description=open("README.md", "r", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    license='MIT',
    url="https://github.com/",
    packages=find_packages(exclude=["*.tests", "*.tests.*",
                                    "tests.*", "tests"]),
    install_requires=['torch>=1.0.0',
                      'numpy',
                      'pandas>=1.0.1',
                      'requests',
                      'tqdm>=4.31.1',
                      'matplotlib',
                      'transformers>=3.0.0'
                      ],
    entry_points={
        'console_scripts': [
            "da-bert-score=da_bert_score_cli.score:main",
        ]
    },
    python_requires='>=3.6'
)
