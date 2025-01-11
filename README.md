# DeepADN
Describe how the acoustic features used by DeepADN were obtained, and the neural network code

featurizer.py defines two classes, AudioFeaturizer and KaldiFbank, for extracting audio features from waveforms, commonly used in speech processing tasks. AudioFeaturizer supports multiple feature extraction methods like MelSpectrogram, MFCC, and Fbank, allowing for flexible feature extraction and normalization. It also handles variable-length input sequences using a masking technique. The KaldiFbank class is a wrapper for Kaldi’s Fbank feature extraction method, which computes filterbank coefficients from the input waveforms. The file provides tools to process and extract meaningful audio features for speech recognition or sound classification tasks.


This config.yml file sets up an audio classification pipeline, specifying dataset handling, preprocessing (using Fbank features), and model training configurations. 

