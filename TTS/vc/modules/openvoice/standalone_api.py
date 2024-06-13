import json
import os

import librosa
import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn

from TTS.vc.modules.openvoice.models import SynthesizerTrn

# vc_checkpoint=model_path, vc_config=config_path, use_cuda=gpu)

# vc_config.audio.output_sample_rate


class custom_sr_config:
    """Class defined to make combatible sampling rate defination with TTS api.py.

    Args:
        sampling rate.
    """

    def __init__(self, value):
        self.audio = self.Audio(value)

    class Audio:
        def __init__(self, value):
            self.output_sample_rate = value


class OpenVoiceSynthesizer(object):
    def __init__(self, vc_checkpoint, vc_config, use_cuda="cpu"):

        if use_cuda:
            self.device = "cuda"
        else:
            self.device = "cpu"

        hps = get_hparams_from_file(vc_config)
        self.vc_config = custom_sr_config(hps.data.sampling_rate)

        # vc_config.audio.output_sample_rate
        self.model = SynthesizerTrn(
            len(getattr(hps, "symbols", [])),
            hps.data.filter_length // 2 + 1,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        ).to(torch.device(self.device))

        self.hps = hps
        self.load_ckpt(vc_checkpoint)
        self.model.eval()

    def load_ckpt(self, ckpt_path):
        checkpoint_dict = torch.load(ckpt_path, map_location=torch.device(self.device))
        a, b = self.model.load_state_dict(checkpoint_dict["model"], strict=False)
        # print("Loaded checkpoint '{}'".format(ckpt_path))
        # print('missing/unexpected keys:', a, b)

    def extract_se(self, fpath):
        audio_ref, sr = librosa.load(fpath, sr=self.hps.data.sampling_rate)
        y = torch.FloatTensor(audio_ref)
        y = y.to(self.device)
        y = y.unsqueeze(0)
        y = spectrogram_torch(
            y,
            self.hps.data.filter_length,
            self.hps.data.sampling_rate,
            self.hps.data.hop_length,
            self.hps.data.win_length,
            center=False,
        ).to(self.device)
        with torch.no_grad():
            g = self.model.ref_enc(y.transpose(1, 2)).unsqueeze(-1)

        return g

    # source_wav="my/source.wav", target_wav="my/target.wav", file_path="output.wav"
    def voice_conversion(self, source_wav, target_wav, tau=0.3, message="default"):

        if not os.path.exists(source_wav):
            print("source wavpath dont exists")
            exit(0)

        if not os.path.exists(target_wav):
            print("target wavpath dont exists")
            exit(0)

        src_se = self.extract_se(source_wav)
        tgt_se = self.extract_se(target_wav)

        # load audio
        audio, sample_rate = librosa.load(source_wav, sr=self.hps.data.sampling_rate)
        audio = torch.tensor(audio).float()

        with torch.no_grad():
            y = torch.FloatTensor(audio).to(self.device)
            y = y.unsqueeze(0)
            spec = spectrogram_torch(
                y,
                self.hps.data.filter_length,
                self.hps.data.sampling_rate,
                self.hps.data.hop_length,
                self.hps.data.win_length,
                center=False,
            ).to(self.device)
            spec_lengths = torch.LongTensor([spec.size(-1)]).to(self.device)
            audio = (
                self.model.voice_conversion(spec, spec_lengths, sid_src=src_se, sid_tgt=tgt_se, tau=tau)[0][0, 0]
                .data.cpu()
                .float()
                .numpy()
            )

        return audio


def get_hparams_from_file(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    return hparams


class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, dict):
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


MAX_WAV_VALUE = 32768.0


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    if torch.min(y) < -1.1:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.1:
        print("max value is ", torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec


def spectrogram_torch_conv(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    # if torch.min(y) < -1.:
    #     print('min value is ', torch.min(y))
    # if torch.max(y) > 1.:
    #     print('max value is ', torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect"
    )

    # ******************** original ************************#
    # y = y.squeeze(1)
    # spec1 = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
    #                   center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    # ******************** ConvSTFT ************************#
    freq_cutoff = n_fft // 2 + 1
    fourier_basis = torch.view_as_real(torch.fft.fft(torch.eye(n_fft)))
    forward_basis = fourier_basis[:freq_cutoff].permute(2, 0, 1).reshape(-1, 1, fourier_basis.shape[1])
    forward_basis = (
        forward_basis * torch.as_tensor(librosa.util.pad_center(torch.hann_window(win_size), size=n_fft)).float()
    )

    import torch.nn.functional as F

    # if center:
    #     signal = F.pad(y[:, None, None, :], (n_fft // 2, n_fft // 2, 0, 0), mode = 'reflect').squeeze(1)
    assert center is False

    forward_transform_squared = F.conv1d(y, forward_basis.to(y.device), stride=hop_size)
    spec2 = torch.stack(
        [forward_transform_squared[:, :freq_cutoff, :], forward_transform_squared[:, freq_cutoff:, :]], dim=-1
    )

    # ******************** Verification ************************#
    spec1 = torch.stft(
        y.squeeze(1),
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )
    assert torch.allclose(spec1, spec2, atol=1e-4)

    spec = torch.sqrt(spec2.pow(2).sum(-1) + 1e-6)
    return spec


def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    global mel_basis
    dtype_device = str(spec.dtype) + "_" + str(spec.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=spec.dtype, device=spec.device)
    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)
    return spec


def mel_spectrogram_torch(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=y.dtype, device=y.device)
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)

    return spec
