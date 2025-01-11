# DeepADN
Describe how the acoustic features used by DeepADN were obtained, and the neural network code

data_utils.featurizer.py defines two classes, AudioFeaturizer and KaldiFbank, for extracting audio features from waveforms, commonly used in speech processing tasks. AudioFeaturizer supports multiple feature extraction methods like MelSpectrogram, MFCC, and Fbank, allowing for flexible feature extraction and normalization. It also handles variable-length input sequences using a masking technique. The KaldiFbank class is a wrapper for Kaldi’s Fbank feature extraction method, which computes filterbank coefficients from the input waveforms. The file provides tools to process and extract meaningful audio features for speech recognition or sound classification tasks.


This config.yml file sets up an audio classification pipeline, specifying dataset handling, preprocessing (using Fbank features), and model training configurations. 

The AudioDetectionnetwork.py file defines a deep neural network for audio classification that combines Convolutional Neural Networks (CNNs) with Channel and Spatial Attention mechanisms (CBAM) to improve feature extraction. The model applies multiple convolutional blocks, followed by fully connected layers, to classify audio input into target classes.

There are some positive and negative sample audio in the Audio folder.
