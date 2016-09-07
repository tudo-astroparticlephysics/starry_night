from distutils.core import setup

setup(
    name='starry_night',
    version='1.0.0',
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
        'setuptools',
        'scikit-image',
        'astropy',
    ],
    test_suite='nose.collector',
    tests_require=['nose'],
    scripts=['scripts/starry_night', 'scripts/configure'],

    package_data={
        'starry_night': [
            'data/catalogue_10vmag_1degFilter.csv',
            'data/CTA_cam.config',
            'data/GTC_cam.config',
            'data/IceCube_cam.config',
            'data/Magic_cam.config',
            'data/example_sources.csv',
            ]
        },
    zip_safe=False
)
