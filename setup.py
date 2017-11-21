from setuptools import setup, find_packages

setup(
    name='starry_night',
    version='1.2.0',
    description='A tool for detecting clouds in the night sky based on star detection.',
    url='https://bitbucket.org/solarer/starry_night',
    author='Jan Adam',
    author_email='jan.adam@tu-dortmund.de',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scipy',
        'pyephem',
        'requests',
        'numpy',
        'matplotlib>=1.4',
        'docopt',
        'setuptools',
        'scikit-image',
        'astropy',
        'tables',
        'configparser',
    ],
    test_suite='nose.collector',
    tests_require=['nose'],
    entry_points={
        'console_scripts': [
            'starry_night = starry_night.scripts.starry_night:main'
        ]
    },
    package_data={
        'starry_night': [
            'data/catalogue_10vmag_1.0degFilter.csv',
            'data/catalogue_10vmag_0.8degFilter.csv',
            'data/CTA_cam.config',
            'data/GTC_cam.config',
            'data/IceCube_cam.config',
            'data/Magic_cam1.config',
            'data/Magic_cam2.config',
            'data/example_sources.csv',
            ]
        },
    zip_safe=False
)
