# Image Filtering #

This is a collection of some common methods for image filtering. 
They have been implemented for educational reasons with a special interest in anisotropic
and isotropic image filtering. Now this package serves as a toolkit 
for working on computer vision tasks.

## Content ##

Currently the following filters are included in ``filters.py``:

- (Isotropic) Perona-Malik-Filter
- (Anisotropic) Coherence Shock Filter by J. Weickert
- Linear Osmosis Filter by J. Weickert

## Usage ##

The properties of each filter are specified by several parameters such as 
kernel sizes, timestep sizes or number of iterations. The choice of them depends on 
the effect one wants to achieve. To help finding the best suiting set of parameters 
one can use the script ``edit.py``.

## Results ##

- Perona-Malik-Filter
![perona-malik-results](Resources/PeronaMalik/sample.png)

 - Coherence Shock Filter
![coherence-results](Resources/Coherence/sample.png)

- Linear Osmosis Filter
![osmosis-results](Resources/Osmosis/sample.png)