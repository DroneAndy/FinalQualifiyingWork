from setuptools import setup, find_packages

import json
import os

def read_pipenv_dependencies(fname):
    """Получаем из Pipfile.lock зависимости по умолчанию."""
    filepath = os.path.join(os.path.dirname(__file__), fname)
    with open(filepath) as lockfile:
        lockjson = json.load(lockfile)
        return [dependency for dependency in lockjson.get('default')]

if __name__ == '__main__':
    setup(
        name='skeleton-converter',
        version=os.getenv('PACKAGE_VERSION', '0.0.dev0'),
        packages=[
            find_packages('skeleton_converter', include=['skeleton_converter', 'skeleton_converter.*']),
            find_packages('posenet', include=['posenet', 'posenet.*']),
            find_packages('alphapose', include=['alphapose', 'alphapose.*']),
        ],
        # packages=['skeleton_converter', 'posenet', 'alphapose'],
        description='Python library for conversion skeleton models to universal model.',
        author='Kovalev Andrey',
        install_requires=[
              *read_pipenv_dependencies('Pipfile.lock'),
        ]
    )
