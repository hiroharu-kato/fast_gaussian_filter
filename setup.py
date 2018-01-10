import setuptools

setuptools.setup(
    name='fast_gaussian_filter',
    version='0.0.1',
    test_suite='tests',
    install_requires=['numpy', 'chainer', 'cupy'],
    packages=['fast_gaussian_filter'],
)