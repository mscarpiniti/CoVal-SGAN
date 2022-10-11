# CoVal-SGAN
This script implements the CoVal-SGAN, a complex-valued GAN architecture working with the audio spectrogram for an effective audio data augmentation proposed in [1].

Specifically, the proposed CoVal-SGAN is used to generate new synthetic spectrogram to augment audio data of the more problematic among the considered class. The augemnted dataset is used for the classification of different equipments on construction sites.

The complex-valued implementation of the proposed GAN exploits the following software package [2], which should be installed. See also: https://github.com/NEGU93/cvnn

The implementation of complex batch normmalization is that proposed in [3]. This implementation is available here: https://github.com/ChihebTrabelsi/deep_complex_networks/blob/master/complexnn/bn.py

1. Michele Scarpiniti, Cristiano Mauri, Danilo Comminiello, Aurelio Uncini, Yong-Cheol Lee, "CoVal-SGAN: A Complex-Valued Spectral GAN architecture for the effective audio data augmentation in construction sites", *2022 International Joint Conference on Neural Networks (IJCNN 2022)*, pp. 1--8, Padova, Italy, 2022, https://doi.org/10.1109/IJCNN55064.2022.9891915.

2. J Agustin Barrachina, "Complex-Valued Neural Networks (CVNN)", v.16, January 2021, Zenodo, https://doi.org/10.5281/zenodo.4452131

3. Chiheb Trabelsi, Olexa Bilaniuk, Ying Zhang, Dmitriy Serdyuk, Sandeep Subramanian, João Felipe Santos, Soroush Mehri, Negar Rostamzadeh, Yoshua Bengio, Christopher J Pal, "Deep Complex Networks", [arXiv:1705.09792](https://arxiv.org/abs/1705.09792), 2017.


## Requirements
To use the CoVal-SGAN, the user, in addition to the normal Python modules (tensorflow, librosa, numpy), must install the cvnn library (https://github.com/NEGU93/cvnn) and download the bn.py file (https://github.com/ChihebTrabelsi/deep_complex_networks/blob/master/complexnn/bn.py). Additional details can be found in the 'requirements.txt' file.
