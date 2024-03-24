from setuptools import setup

setup(
    name='GradCache',
    version='0.1.0',
    packages=['grad_cache', 'grad_cache.cachex', 'grad_cache.pytorch_lightning'],
    package_dir={'': 'src', 'grad_cache': 'src/grad_cache'},
    url='https://github.com/luyug/GradCache',
    license='Apache-2.0',
    author='Luyu Gao',
    author_email='luyug@cs.cmu.edu',
    description=''
)
