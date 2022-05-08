# CoVal-SGAN
This script implements the CoVal-SGAN, a complex-valued GAN architecture working with the audio spectrogram for an effective audio data augmentation proposed in:

<a id="1">[1]</a> 
Michele Scarpiniti, Cristiano Mauri, Danilo Comminiello, Aurelio Uncini, CoVal-SGAN: A Complex-Valued Spectral GAN architecture for the effective audio data augmentation in construction sites, *2022 International Joint Conference on Neural Networks (IJCNN 2022)*, pp. 1--8, Padova, Italy, 2022.

Specifically, the proposed CoVal-SGAN is used to generate new synthetic spectrogram to augment audio data of the more problematic among the considered class. The augemnted dataset is used for the classification of different equipments on construction sites.

The complex-valued implementation of the proposed GAN exploits the following software package, which should be installed:

@SOFTWARE {j_agustin_barrachina_2021_4452131,
  author       = "J Agustin Barrachina",
  title        = "Complex-Valued Neural Networks (CVNN)",
  month        = "jan",
  year         = "2021",
  publisher    = "Zenodo",
  version      = "v1.6",
  doi          = "10.5281/zenodo.4452131",
  url          = "https://doi.org/10.5281/zenodo.4452131"
}

See also: https://github.com/NEGU93/cvnn


The implementation of complex batch normmalization is that proposed in:
    
    @ARTICLE {Trabelsi2017,
    author  = "Chiheb Trabelsi, Olexa Bilaniuk, Ying Zhang, Dmitriy Serdyuk, Sandeep Subramanian, João Felipe Santos, Soroush Mehri, Negar Rostamzadeh, Yoshua Bengio, Christopher J Pal",
    title   = "Deep Complex Networks",
    journal = "arXiv preprint arXiv:1705.09792",
    year    = "2017"
}
    
This implementation is available here: https://github.com/ChihebTrabelsi/deep_complex_networks/blob/master/complexnn/bn.py

