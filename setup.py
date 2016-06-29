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
        #'pkg_resources',
        'configparser',
        'scikit-image',
        'astropy',
    ],
    test_suite='nose.collector',
    tests_require=['nose'],
    scripts=['scripts/starry_night'],

    package_data={
        'starry_night': [
            'data/catalogue_10vmag_1degFilter.csv',
            'CTA_cam.config',
            'GTC_cam.config',
            'IceCube_cam.config',
            'Magic_cam.config',
            ]
        },
    zip_safe=False
)
