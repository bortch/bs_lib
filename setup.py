from setuptools import find_packages, setup
setup(
    name='bs_lib',
    packages=find_packages(include=['bs_lib']),
    version='0.1.0',
    description='ML library',
    author='BS',
    license='MIT',
    install_requires=['tensorflow','pandas','numpy','scipy','sklearn','matplotlib','seaborn','rich'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='tests',
)