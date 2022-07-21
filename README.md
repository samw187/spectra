


# Converting Multi-Colour Images into Spectra Using Deep Learning

This repository includes the main python scripts used to complete my Master's project at Durham University. 
This project aimed to create a pipeline which could take galaxy image data as an input and output galaxy spectra..

## Description

In this work, we use deep learning techniques and generative modelling to produce galactic spectra from 5-band “grizy” optical images. A Variational Autoencoder (VAE) is used firstly to encode spectra into a low
dimensional latent distribution, and these latent features are used as “labels” for a Convolutional Neural Network (CNN). The CNN is then trained to predict latent features from
photometric data.
Candidate galaxies were initially selected from the SDSS imaging survey with an upper limit magnitude set
at g < 18. Spectral data were then collected from the SDSS (DR16) and normalised. Imaging data (as well as
spectral normalisation factors) were collected from Pan-STARRS to give a final data set of
around 36000 galaxies. The pipeline to transfer multi-colour images into
spectra is complete with an overall average deviation of 10.2 ± 0.6 SDSS spectral errors
compared to the target spectra (averaged across 7579 spectra). All models were trained
using keras and tensorflow on an NVIDIA Tesla V100-PCIE-32GB GPU. 

## Motivation

A spectrum takes 1000 times longer to produce than an image and upcoming astronomical surveys such as Euclid or the Large Synoptic Sky Telescope survey will include no optical spectroscopy. 
These surveys will encompass hundreds of millions of images of objects, hence it is unfeasible to gather the spectra with direct observations in a follow up study, so a tool which can easily convert between images and spectra would allow for a better match up in the volumes of the two data types. 
Spectral features such as absorption or emission lines give information about the interstellar medium and stellar types in a galaxy which cannot be immediately inferred from photometric data.

## Getting Started

### Dependencies

All model scripts require the TensorFlow and Keras modules, whilst data scripts require Astropy and Linetools. 

An initial selection of galaxies to use with this project can be made using the following search form:

http://skyserver.sdss.org/dr16/en/tools/search/form/searchform.aspx

### Sub-directories

The listed folders in this repository are as follows.

CNN_and_VAE_testing: Contains the main python scripts used to train and optimise the CNN and VAE models used in this project. 

Database_building_files: Contains the main python scripts used to gather the photometric and spectral data in this project, as well as normalisation factors. Note that to use these scripts,
a .csv file of candidate galaxies must first be downloaded using the search form in the "Dependencies" section. 
 
Legacy_data_and_normalisation_files: Other database building files using older methods. These scripts were not integral to the final project, but are included in the repository as a record of previous methods. 

Legacy_data_npz_files_redundant: Old data files saved to this repository as a record. Not required. 

Older_Models: Other old CNN and VAE scripts containing trials with different types of models. Again, saved to the repository as a record but these were not part of the final pipeline of the project.


### Executing program

1) Download a database of galaxies from the SDSS using the search form.

2) Download corresponding spectral data using Spectra_getter.py

3) Download corresponding image data using get_fits_images_redone.py

4) Normalisation factors can be found using Gather_normalisation_values.py

5) Both the spectral data and the normalisation factors for the spectra should be matched using RA and Dec coordinates (these coordinates are saved to both the normalisation file, and the spectral database).
No script is included for this part but the process is straightforward.

5) A VAE can be optimised and trained using VAEtester.py (this also saves the corresponding latent variable representations of the galaxy spectra)

6) Match_images_to_latent_vars.py can be used to couple latent variable representations of spectra to the galaxy image data. This also processes the image dataset to remove any bad data.

7) Use CNN_trainer.py to firstly pre-process and augment the image data, then to train and optimise a CNN model. 


## Acknowledgments

For their excellent supervision throughout this project, this work acknowledges Dr Ryan J. Cooke and Dr Ting-Yun Cheng, as well as the Durham University Physics department. 

Furthermore, inspiration for this project came from two research papers: 

Wu, J.F. and Peek, J.E., (2020). Predicting galaxy spectra from images with hybrid convolutional
neural networks. arXiv preprint arXiv:2009.12318.

Portillo, S.K.N. et al. (2020). Dimensionality Reduction of SDSS Spectra with Variational Autoencoders. The Astronomical Journal, 160(1), p.45.


