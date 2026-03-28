"""
futureQuant 期货量化研究框架
"""

from setuptools import setup, find_packages
from pathlib import Path

# 读取README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding='utf-8') if readme_file.exists() else ""

# 读取requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="futureQuant",
    version="0.1.0",
    author="futureQuant Team",
    description="期货量化研究框架 - 支持数据管理、因子分析、策略回测、模型训练",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.0.0',
        ],
        'ml': [
            'torch>=2.0.0',
            'transformers>=4.30.0',
        ],
        'crawler': [
            'playwright>=1.40.0',
            'DrissionPage>=4.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'futureQuant=futureQuant.main:main',
        ],
    },
    include_package_data=True,
    package_data={
        'futureQuant': ['config/*.yaml', 'config/varieties/*.yaml'],
    },
)
