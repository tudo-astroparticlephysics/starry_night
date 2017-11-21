
# Starry_night

A python command line tool to analyze the cloudiness of the night sky using images taken by a fish-eye lens.

## Getting Started


### Prerequisites

To run Starry_Night you need a working Python 3 installation as well as image files (from hard disk or from a website)
and a valid configuration file that takes into account the properties of your camera. 

A few example configuration files are provided within the package, these are set up to download real-time 
images from cameras set up on Roque de los Muchachos on La Palma.

### Installing

Just run
```
pip install git+https://github.com/tudo-astroparticlephysics/starry_night.git
```
and pip does all the magic for you.


## Running Starry_Night

Starry_Night is a command line tool and has no GUI. You can call
```
starry_night -h
```

to get a list of all valid parameters for running Starry_Night. A few of them are also explained further below.

The most basic example of running Starry_Night is

```
starry_night -c CAM_NAME -v --cam
```
with CAM_NAME being the name of your camera configuration file you want to use.

To begin with, you can choose from "cta", "magic" and "gtc" as CAM_NAME. These config files are set up to automatically download a real-time image so you do not need to provide images yourself.
Unless you pass an image to Starry_Night it will always try to download an image from the website mentioned in the config file.

You can also pass a list of images or a whole directory to Starry_Night. It will also search within subfolders.
```
starry_night -c CAM_NAME /path/to/image1.png /path/to/image_dir/ -v --cam
```

## Program flags explained

### Configuration
```
-c CONFIG_FILE
```
Passing a configuration file to Starry_Night is mandatory! You can pass the full path to a config file or just
the beginning of the camera name if your config file is located in starry_night/starry_night/data and is named like this: [CAMERA_NAME]_cam.config.
You can load multiple config files at once if they share the same name: e.g. gtc_cam.config, gtc_cam_version2.config, ...
this is useful if you analyze a series of images and at some point in time a parameter of your camera changed (camera was moved/turned). The value of "useConfAfter" in the config file will determine which config file should process the current image.

### Visualization

You don't always want to see the results of Starry_Night because you might process a lot of images and you only want to store the results in a database for further processing. Because of that visual output is deactivated by default!
Use the following 2 flags to change this behaviour and also choose one of --cam, --response, ... to decide on WHAT you want to output. Starry_Night saves computation time by not calculating stuff you do not want.
```
-v and -s
```
The -v flag turns on visual output of images. Keep it off if you are only interested in the data and don't want to look at every image/plot right now or if you run this as a daemon at night.
The -s flag turns on the save function. All generated images/plots will be saved to disk with a timestamp when using this flag.
You can combine both flags if you want.

```
--cam
```

Draw the current image with visible and covered stars. Use -v to output or -s to save this or you won't see anything.

```
--response
```

Calculate the kernel response for every star in the image. Don't forget to combine this with -v or -s.

```
--cloudmap
```
Generate a cloud map for the current image.

```
--cloudtrack
```
Try to predict speed and direction of clouds as they move by. This requires a series of at least 2 consecutive images.
This feature was only included as a proof of concept and needs significant improvement.

```
--ratescan
```
- Currently broken - 
Should generate a ratescan-like plot to compare the effectiveness of different image kernels for detecting stars. Needs to be reworked from ground.

## Troubleshooting

If you do not have an xserver running, execute 
```
export MPLBACKEND="agg"
```
in console before running Starry_Night. You will also need -s option to get output in form of files because -v will not work.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

