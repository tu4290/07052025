from setuptools import setup, find_packages

setup(
    name="huihui_integration",
    version="0.1.0",
    packages=find_packages(),
    author="Cascade",
    author_email="",
    description="Elite Options System v2.5 with HuiHui Integration",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'aiohttp',
        'prometheus-client',
    ],
)
