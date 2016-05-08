from distutils.core import setup

setup(
    name='starry_sky',
    version='0.0.1',
    description='A tool for calculating the cloudiness of the night sky based on star detection',
    url='https://bitbucket.org/solarer/starry_night',
    author='Jan Adam',
    author_email='jan.adam@tu-dortmund.de',
    license='MIT',
    packages=[
        'starry_sky',
        ],
    install_requires=[
        'pandas',           # in anaconda
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
    #scripts=['scripts/shift_helper', 'scripts/qla_bot'],
    #package_data={'fact_shift_helper.tools': ['config.gpg']},
    zip_safe=False
)
