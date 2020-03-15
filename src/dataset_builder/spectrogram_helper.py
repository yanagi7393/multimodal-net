import numpy as np
from phase_helper import instantaneous_frequency


# mel spectrum constants.
_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0


def mel_to_hertz(mel_values):
    """Converts frequencies in `mel_values` from the mel scale to linear scale."""
    return _MEL_BREAK_FREQUENCY_HERTZ * (
        np.exp(np.array(mel_values) / _MEL_HIGH_FREQUENCY_Q) - 1.0
    )


def hertz_to_mel(frequencies_hertz):
    """Converts frequencies in `frequencies_hertz` in Hertz to the mel scale."""
    return _MEL_HIGH_FREQUENCY_Q * np.log(
        1.0 + (np.array(frequencies_hertz) / _MEL_BREAK_FREQUENCY_HERTZ)
    )


def linear_to_mel_weight_matrix(
    num_mel_bins=20,
    num_spectrogram_bins=129,
    sample_rate=16000,
    lower_edge_hertz=125.0,
    upper_edge_hertz=3800.0,
):
    """Returns a matrix to warp linear scale spectrograms to the mel scale.
  Adapted from tf.contrib.signal.linear_to_mel_weight_matrix with a minimum
  band width (in Hz scale) of 1.5 * freq_bin. To preserve accuracy,
  we compute the matrix at float64 precision and then cast to `dtype`
  at the end. This function can be constant folded by graph optimization
  since there are no Tensor inputs.
  Args:
    num_mel_bins: Int, number of output frequency dimensions.
    num_spectrogram_bins: Int, number of input frequency dimensions.
    sample_rate: Int, sample rate of the audio.
    lower_edge_hertz: Float, lowest frequency to consider.
    upper_edge_hertz: Float, highest frequency to consider.
  Returns:
    Numpy float32 matrix of shape [num_spectrogram_bins, num_mel_bins].
  Raises:
    ValueError: Input argument in the wrong range.
  """
    # Validate input arguments
    if num_mel_bins <= 0:
        raise ValueError("num_mel_bins must be positive. Got: %s" % num_mel_bins)
    if num_spectrogram_bins <= 0:
        raise ValueError(
            "num_spectrogram_bins must be positive. Got: %s" % num_spectrogram_bins
        )
    if sample_rate <= 0.0:
        raise ValueError("sample_rate must be positive. Got: %s" % sample_rate)
    if lower_edge_hertz < 0.0:
        raise ValueError(
            "lower_edge_hertz must be non-negative. Got: %s" % lower_edge_hertz
        )
    if lower_edge_hertz >= upper_edge_hertz:
        raise ValueError(
            "lower_edge_hertz %.1f >= upper_edge_hertz %.1f"
            % (lower_edge_hertz, upper_edge_hertz)
        )
    if upper_edge_hertz > sample_rate / 2:
        raise ValueError(
            "upper_edge_hertz must not be larger than the Nyquist "
            "frequency (sample_rate / 2). Got: %s for sample_rate: %s"
            % (upper_edge_hertz, sample_rate)
        )

    # HTK excludes the spectrogram DC bin.
    bands_to_zero = 1
    nyquist_hertz = sample_rate / 2.0
    linear_frequencies = np.linspace(0.0, nyquist_hertz, num_spectrogram_bins)[
        bands_to_zero:, np.newaxis
    ]
    # spectrogram_bins_mel = hertz_to_mel(linear_frequencies)

    # Compute num_mel_bins triples of (lower_edge, center, upper_edge). The
    # center of each band is the lower and upper edge of the adjacent bands.
    # Accordingly, we divide [lower_edge_hertz, upper_edge_hertz] into
    # num_mel_bins + 2 pieces.
    band_edges_mel = np.linspace(
        hertz_to_mel(lower_edge_hertz), hertz_to_mel(upper_edge_hertz), num_mel_bins + 2
    )

    lower_edge_mel = band_edges_mel[0:-2]
    center_mel = band_edges_mel[1:-1]
    upper_edge_mel = band_edges_mel[2:]

    freq_res = nyquist_hertz / float(num_spectrogram_bins)
    freq_th = 1.5 * freq_res
    for i in range(0, num_mel_bins):
        center_hz = mel_to_hertz(center_mel[i])
        lower_hz = mel_to_hertz(lower_edge_mel[i])
        upper_hz = mel_to_hertz(upper_edge_mel[i])
        if upper_hz - lower_hz < freq_th:
            rhs = 0.5 * freq_th / (center_hz + _MEL_BREAK_FREQUENCY_HERTZ)
            dm = _MEL_HIGH_FREQUENCY_Q * np.log(rhs + np.sqrt(1.0 + rhs ** 2))
            lower_edge_mel[i] = center_mel[i] - dm
            upper_edge_mel[i] = center_mel[i] + dm

    lower_edge_hz = mel_to_hertz(lower_edge_mel)[np.newaxis, :]
    center_hz = mel_to_hertz(center_mel)[np.newaxis, :]
    upper_edge_hz = mel_to_hertz(upper_edge_mel)[np.newaxis, :]

    # Calculate lower and upper slopes for every spectrogram bin.
    # Line segments are linear in the mel domain, not Hertz.
    lower_slopes = (linear_frequencies - lower_edge_hz) / (center_hz - lower_edge_hz)
    upper_slopes = (upper_edge_hz - linear_frequencies) / (upper_edge_hz - center_hz)

    # Intersect the line segments with each other and zero.
    mel_weights_matrix = np.maximum(0.0, np.minimum(lower_slopes, upper_slopes))

    # Re-add the zeroed lower bins we sliced out above.
    # [freq, mel]
    mel_weights_matrix = np.pad(
        mel_weights_matrix, [[bands_to_zero, 0], [0, 0]], "constant"
    )
    return mel_weights_matrix


def _linear_to_mel_matrix():
    """Get the mel transformation matrix."""
    _sample_rate = 16000
    _mel_downscale = 1
    num_freq_bins = 2048 // 2
    lower_edge_hertz = 0.0
    upper_edge_hertz = 16000 / 2.0
    num_mel_bins = num_freq_bins // _mel_downscale
    return linear_to_mel_weight_matrix(
        num_mel_bins, num_freq_bins, _sample_rate, lower_edge_hertz, upper_edge_hertz
    )


def _mel_to_linear_matrix():
    """Get the inverse mel transformation matrix."""
    m = _linear_to_mel_matrix()
    m_t = np.transpose(m)
    p = np.matmul(m, m_t)
    d = [1.0 / x if np.abs(x) > 1.0e-8 else x for x in np.sum(p, axis=0)]
    return np.matmul(m_t, np.diag(d))


def melspecgrams_to_specgrams(logmelmag2, mel_p):
    """Converts melspecgrams to specgrams.
    Args:
      melspecgrams: Tensor of log magnitudes and instantaneous frequencies,
        shape [freq, time], mel scaling of frequencies.
    Returns:
      specgrams: Tensor of log magnitudes and instantaneous frequencies,
        shape [freq, time].
    """
    logmelmag2 = logmelmag2.T
    mel_p = mel_p.T
    logmelmag2 = np.array([logmelmag2])
    mel_p = np.array([mel_p])

    mel2l = _mel_to_linear_matrix()
    mag2 = np.tensordot(np.exp(logmelmag2), mel2l, 1)
    logmag = 0.5 * np.log(mag2 + 1e-6)
    mel_phase_angle = np.cumsum(mel_p * np.pi, axis=1)
    phase_angle = np.tensordot(mel_phase_angle, mel2l, 1)
    p = instantaneous_frequency(phase_angle, time_axis=1)
    return logmag[0].T, p[0].T


def specgrams_to_melspecgrams(magnitude, IF):
    """Converts specgrams to melspecgrams.
    Args:
      specgrams: Tensor of log magnitudes and instantaneous frequencies,
        shape [freq, time].
    Returns:
      melspecgrams: Tensor of log magnitudes and instantaneous frequencies,
        shape [freq, time], mel scaling of frequencies.
    """
    logmag = magnitude.T
    p = IF.T
    mag2 = np.exp(2.0 * logmag)
    mag2 = np.array([mag2])
    phase_angle = np.cumsum(p * np.pi, axis=1)
    phase_angle = np.array([phase_angle])

    l2mel = _linear_to_mel_matrix()
    logmelmag2 = np.log(np.tensordot(mag2, l2mel, axes=1) + 1e-6)
    mel_phase_angle = np.tensordot(phase_angle, l2mel, axes=1)
    mel_p = instantaneous_frequency(mel_phase_angle, time_axis=1)
    return logmelmag2[0].T, mel_p[0].T
