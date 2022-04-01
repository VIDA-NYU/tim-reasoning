import os
import setuptools


def read_readme():
    with open(os.path.join(os.path.dirname(__file__), 'README.md'), encoding='utf8') as file:
        return file.read()


def read_version():
    module_path = os.path.join('tim_reasoning', '__init__.py')
    with open(module_path) as file:
        for line in file:
            parts = line.strip().split(' ')
            if parts and parts[0] == '__version__':
                return parts[-1].strip("'")

    raise KeyError('Version not found in {0}'.format(module_path))


package_name = 'tim-reasoning'
long_description = read_readme()
version = read_version()

with open('requirements.txt') as fp:
    req = [line for line in fp if line and not line.startswith('#')]


setuptools.setup(
    name=package_name,
    version=version,
    packages=setuptools.find_packages(),
    install_requires=req,
    description="Reasoning module for TIM",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/VIDA-NYU/tim-reasoning',
    include_package_data=True,
    author='Roque Lopez',
    author_email='rlopez@nyu.edu',
    maintainer='Roque Lopez',
    maintainer_email='rlopez@nyu.edu',
    keywords=['reasoning', 'ptg', 'nyu'],
    license='Apache-2.0',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering',
    ])
