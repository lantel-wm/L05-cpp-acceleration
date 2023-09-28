from setuptools import setup, find_packages

setup(
    name="lorenz05-python",
    version="1.0",
    author="Zhiyu Zhao",
    author_email="zyzh@smail.nju.edu.cn",
    description="Lorenz 05 model in Python",
    packages=find_packages(),
    
    classifiers= [
        "License :: OSI Approved :: MIT License",
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.10',
    ],
    
    data_files=[('', ['*.ini'])],
    
    install_requires=[
        'numpy>=1.22.4',
        'matplotlib',
    ]
    
    
)