# Starry_night

Detailed README file will be provided soon.

Installation: pip install git+https://github.com/tudo-astroparticlephysics/starry_night.git

Basic usage at night: starry_night -c [Magic or CTA] --cam -v
to download and see the current all sky camera image. Other options such as --cloudmap and --response are available, too.

Option -s stores images as png file.

Option --help lists all available comments.

If you dont have an xserver, execute: export MPLBACKEND="agg" in console before running starry_night. You will also need -s option to get output in form of files.
