from setuptools import setup, find_packages

setup(
    name="assmanager",
    version="1.0.0",
    author="Zhiyu Zhao",
    author_email="zyzh@smail.nju.edu.cn",
    description="Lorenz 05 model in Python",
    packages=find_packages(),
    package_dir={'assmanager': 'assmanager'},
    url='https://github.com/ZZY000926/assmanager.git',
    classifiers= [
        "License :: OSI Approved :: MIT License",
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.8',
    # data_files=[
    #     ('assmanager', ['assmanager/config.ini']),
    #     ('assmanager', ['assmanager/template.ini']),
    #     ],
    install_requires=[
        'numba',
        'numpy',
        'configparser',
        'scipy',
        'tqdm',
    ]
    
)