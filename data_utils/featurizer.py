import torch
import torchaudio.compliance.kaldi as Kaldi
from torch import nn
from torchaudio.transforms import MelSpectrogram, MFCC


class AudioFeaturizer(nn.Module):
    

    def __init__(self, feature_method='MelSpectrogram', method_args={}):
        super().__init__()
        self._method_args = method_args
        self._feature_method = feature_method
        if feature_method == 'MelSpectrogram':
            self.feat_fun = MelSpectrogram(**method_args)
        elif feature_method == 'MFCC':
            self.feat_fun = MFCC(**method_args)
        elif feature_method == 'Fbank':
            self.feat_fun = KaldiFbank(**method_args)
        else:
            raise Exception(f'The preprocessing method {self. _feature_math} does not exist!')

    def forward(self, waveforms, input_lens_ratio=None):
        """Extracting audio features from AudioSegment

        :param waveforms: Audio segment to extract features from.
        :type waveforms: AudioSegment
        :param input_lens_ratio: input length ratio
        :type input_lens_ratio: tensor
        :return: Spectrogram audio feature in 2darray.
        :rtype: ndarray
        """
        if len(waveforms.shape) == 1:
            waveforms = waveforms.unsqueeze(0)
        feature = self.feat_fun(waveforms)
        feature = feature.transpose(2, 1)
        # normalization
        feature = feature - feature.mean(1, keepdim=True)
        if input_lens_ratio is not None:
            input_lens = (input_lens_ratio * feature.shape[1])
            mask_lens = torch.round(input_lens).long()
            mask_lens = mask_lens.unsqueeze(1)
            idxs = torch.arange(feature.shape[1], device=feature.device).repeat(feature.shape[0], 1)
            mask = idxs < mask_lens
            mask = mask.unsqueeze(-1)
            feature = torch.where(mask, feature, torch.zeros_like(feature))
        return feature

    
    def feature_dim(self):

        if self._feature_method == 'MelSpectrogram':
            n_mels = self._method_args.get('n_mels', 40)  
            return n_mels  
        elif self._feature_method == 'MFCC':
            return self._method_args.get('n_mfcc', 40)
        elif self._feature_method == 'MFCC':
            n_mfcc = self._method_args.get('n_mfcc', 40)  
            melkwargs = {
                'n_mels': self._method_args.get('n_mels', 40),
                'f_min': self._method_args.get('f_min', 0),     
                'f_max': self._method_args.get('f_max', 3000)     
                  }
            return n_mfcc, melkwargs  
        elif self._feature_method == 'Fbank':
            return self._method_args.get('num_mel_bins', 40)
        else:
            raise Exception('There is no {} preprocessing method'. format (self. _feature_math))


class KaldiFbank(nn.Module):
    def __init__(self, **kwargs):
        super(KaldiFbank, self).__init__()
        self.kwargs = kwargs

    def forward(self, waveforms):
        """
        :param waveforms: [Batch, Length]
        :return: [Batch, Length, Feature]
        """
        log_fbanks = []
        for waveform in waveforms:
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)
            log_fbank = Kaldi.fbank(waveform, **self.kwargs)
            log_fbank = log_fbank.transpose(0, 1)
            log_fbanks.append(log_fbank)
        log_fbank = torch.stack(log_fbanks)
        return log_fbank
