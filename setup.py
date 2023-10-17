from setuptools import setup, find_packages, Extension
import os

def get_so_filepath():
    so_dir = './assmanager/model/step_L04/cpu_parallel/'
    for file in os.listdir(so_dir):
        if file.endswith('.so'):
            return os.path.join(so_dir, file)

setup(
    name = "assmanager",
    version = "1.0.2",
    author = "Zhiyu Zhao",
    author_email = "zyzh@smail.nju.edu.cn",
    description = "Lorenz 05 model in Python",
    packages = find_packages(),
    package_dir = {'assmanager': 'assmanager'},
    url='https://github.com/ZZY000926/assmanager.git',
    package_data = {'assmanager': [
        './assmanager/model/step_L04/cpu_parallel/*.so',
        ]},
    # data_files = [('assmanager', [get_so_filepath()])],
    classifiers = [
        "License :: OSI Approved :: MIT License",
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires = '>=3.8',
    install_requires = [
        'numba',
        'numpy',
        'configparser',
        'scipy',
        'tqdm',
        'pybind11'
    ],
)