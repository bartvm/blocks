import blocks
from codecs import open
from os import path
from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    while not f.readline().startswith('Blocks'):  # Skip the badges
        pass
    long_description = 'Blocks\n' + f.read().strip()

exec_results = {}
with open(path.join(path.dirname(__file__), 'blocks/version.py')) as file_:
    exec(file_.read(), exec_results)
version = exec_results['version']

setup(
    name='blocks',
    version=version,
    description='A Theano framework for building and training neural networks',
    long_description=long_description,
    url='https://github.com/mila-udem/blocks',
    author='University of Montreal',
    license='MIT',
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
    ],
    keywords='theano machine learning neural networks deep learning',
    packages=find_packages(exclude=['examples', 'docs', 'doctests', 'tests']),
    scripts=['bin/blocks-continue'],
    setup_requires=['numpy'],
    install_requires=['numpy', 'six', 'pyyaml', 'toolz', 'theano',
                      'picklable-itertools', 'progressbar2', 'fuel'],
    extras_require={
        'test': ['mock', 'nose', 'nose2'],
        'docs': ['sphinx', 'sphinx-rtd-theme']
    },
    zip_safe=False)
