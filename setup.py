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
        'pandas',           # in anaconda
        'scipy',            # in anaconda
        'pyephem',            # in anaconda
        #'requests',         # in anaconda
        'numpy',            # in anaconda
        'matplotlib>=1.4',  # in anaconda
        #'python-dateutil',  # in anaconda
        #'sqlalchemy',       # in anaconda
        #'PyMySQL',          # in anaconda
        #'pytz',             # in anaconda
        #'blessings',
        'docopt',           # in anaconda
    ],
    test_suite='nose.collector',
    tests_require=['nose'],
    scripts=['scripts/starry_night'],

    package_data={'starry_night': ['data/*.csv']},
    zip_safe=False
)
