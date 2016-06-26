from distutils.core import setup

setup(
    name='starry_night',
    version='0.0.1',
    description='A tool for calculating the cloudiness of the night sky based on star detection',
    url='https://bitbucket.org/solarer/starry_night',
    author='Jan Adam',
    author_email='jan.adam@tu-dortmund.de',
    license='MIT',
    packages=[
        'starry_night',
        ],
    install_requires=[
        'pandas',           
        'scipy',           
        'pyephem',        
        'requests',      
        'numpy',        
        'matplotlib>=1.4',
        'docopt',        
        'pkg_resources',
        'logging',
        'os',
        'sys',
        'time',
        'datetime',
        'configparser',
        'skimage',
        'multiprocessing',
        'functools',
        'astropy',
    ],
    test_suite='nose.collector',
    tests_require=['nose'],
    scripts=['scripts/starry_night'],

    package_data={'starry_night': ['data/asu.tsv']},
    zip_safe=False
)
