import os
import math

import librosa
import matplotlib.pyplot as plt
import pesto
import torch
import torchaudio

from silero_vad import load_silero_vad, get_speech_timestamps

from songify import utils


def next_power_of_two(n):
    """Return the next power of two greater than or equal to n."""
    return 2 ** math.ceil(math.log2(n))


def estimate_pitch(
    audio: torch.Tensor,
    sample_rate: int,
    frame_size_millis: int = 10,
    pitch_strategy: str = "pesto",
):
    if pitch_strategy == "pesto":
        timesteps, pitches, confidence, _ = pesto.predict(
            audio, sample_rate, step_size=frame_size_millis
        )
        timesteps = timesteps / 1000.0
    elif pitch_strategy == "librosa":
        frame_samples = int(sample_rate * frame_size_millis / 1000.0)
        pitches, _, confidence = librosa.pyin(
            audio.numpy(),
            fmin=librosa.note_to_hz("C3"),
            fmax=librosa.note_to_hz("C7"),
            sr=sample_rate,
            frame_length=frame_samples,
            hop_length=frame_samples,
            fill_na=False,
        )
        timesteps = torch.arange(len(pitches)) * (frame_size_millis / 1000.0)
        pitches = torch.from_numpy(pitches)
        confidence = torch.from_numpy(confidence / confidence.max())
    else:
        raise NotImplementedError(
            f"Pitch strategy '{pitch_strategy}' is not implemented."
        )

    pitches = torch.round(69 + 12 * torch.log2(pitches / 440.0)).int()

    return timesteps, pitches, confidence


def get_rms_energy(
    audio: torch.Tensor,
    sample_rate: int,
    frame_size_millis: int = 10,
    window: str = "rectangular",
):
    """
    Calculate the RMS energy of the audio signal in frames.

    Args:
        audio (torch.Tensor): Audio signal.
        sample_rate (int): Sample rate of the audio.
        frame_size_millis (int): Size of each frame in milliseconds.
        window (str): Type of window to apply. Currently only 'rectangular' is supported.

    Returns:
        torch.Tensor: RMS energy of the audio signal in frames.
    """
    frame_size_samples = int(sample_rate * frame_size_millis / 1000.0)

    window = window.lower()
    if window == "rectangular":
        window_fn = torch.ones((1, 1, frame_size_samples))
    else:
        raise NotImplementedError(f"Window type '{window}' is not implemented.")

    squared_audio = audio**2
    windowed_power = torch.conv1d(
        squared_audio.view((1, 1, -1)),
        window_fn,
        stride=frame_size_samples,
        padding=frame_size_samples // 2,
    ).view(-1)
    rms = torch.sqrt(windowed_power)
    return rms


def get_vad(
    audio: torch.Tensor,
    sample_rate: int,
    frame_size_millis: int = 10,
    strategy: str = "default",
    min_voice_duration_millis: int = 100,
    max_voice_duration_millis: int = 2000,
    offset_relative_threshold_db: float = -6.0,
    offset_absolute_threshold_db: float = -48.0,
):
    """
    Voice Activity Detection (VAD) to filter out silent parts of the audio.

    Args:
        audio (torch.Tensor): Audio signal.
        sample_rate (int): Sample rate of the audio.
        frame_size_millis (int): Size of each frame in milliseconds.
        threshold (float): Energy threshold for voice offset estimation.
        strategy (str): Strategy for VAD, currently only 'default' is implemented.

    Returns:
        torch.Tensor: Filtered audio signal with silent parts removed.
    """
    frame_size_samples = int(sample_rate * frame_size_millis / 1000.0)
    voice_onsets_seconds = None

    rms = get_rms_energy(audio, sample_rate, frame_size_millis, window="rectangular")
    rms_max_pool_kernel_size = 5
    rms_max_pooled = torch.nn.functional.max_pool1d(
        rms.view(1, 1, -1),
        kernel_size=rms_max_pool_kernel_size,
        stride=1,
        padding=rms_max_pool_kernel_size // 2,
    ).view(-1)
    rms_max_pooled = torch.roll(rms_max_pooled, rms_max_pool_kernel_size // 2)
    rms_max_pooled[:rms_max_pool_kernel_size // 2] = 0

    rms_absolute_threshold = librosa.db_to_amplitude(offset_absolute_threshold_db).item()
    rms_relative_threshold = librosa.db_to_amplitude(offset_relative_threshold_db).item()

    if strategy == "rms_energy":
        voice_activity = (rms > rms_absolute_threshold) & (
            rms > rms_max_pooled * rms_relative_threshold
        )
        return voice_activity

    elif strategy == "librosa":
        voice_onsets_seconds = librosa.onset.onset_detect(
            y=audio.numpy(),
            sr=sample_rate,
            units="time",
            backtrack=True,
            hop_length=frame_size_samples,
        ).tolist()

    elif strategy == "rms_flux":
        # Calculate RMS flux
        rms_flux = torch.roll(rms, -1) - rms
        rms_flux[-1] = 0
        rectified_rms_flux = torch.maximum(rms_flux, torch.zeros_like(rms_flux))
        rectified_rms_flux = rectified_rms_flux / rectified_rms_flux.max()
        peaks = librosa.util.peak_pick(
            rectified_rms_flux.numpy(),
            pre_max=3,
            post_max=3,
            pre_avg=3,
            post_avg=3,
            delta=0.001,
            wait=int(min_voice_duration_millis / frame_size_millis),
        )
        voice_onsets_seconds = [
            peak * frame_size_millis / 1000.0 for peak in peaks
        ]

    elif strategy == "silero":
        model = load_silero_vad()

        # model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True)
        # (get_speech_timestamps,_, read_audio, *_) = utils
        target_sample_rate = 16000
        if sample_rate != target_sample_rate:
            resampled_audio = torchaudio.functional.resample(
                audio, orig_freq=sample_rate, new_freq=target_sample_rate
            )
        voice_timestamps = get_speech_timestamps(
            resampled_audio,
            model,
            sampling_rate=target_sample_rate,
            return_seconds=True,
        )
        # Convert voice_onsets to a boolean mask
        voice_activity = torch.zeros_like(rms, dtype=torch.bool)
        for speech in voice_timestamps:
            start_frame = int(speech["start"] * 1000.0 / frame_size_millis)
            end_frame = int(speech["end"] * 1000.0 / frame_size_millis)
            voice_activity[start_frame:end_frame] = True
        return voice_activity

    else:
        raise NotImplementedError(f"VAD strategy '{strategy}' is not implemented.")

    if voice_onsets_seconds is None:
        raise ValueError(
            "No voice onsets detected. Check the audio input or VAD strategy."
        )

    # Convert voice_onsets to a boolean mask
    voice_activity = torch.zeros_like(rms, dtype=torch.bool)

    for onset in voice_onsets_seconds:
        start_frame = int(onset * sample_rate / frame_size_millis)
        end_frame = start_frame + 1
        while end_frame < len(rms) and rms[end_frame] > rms_absolute_threshold:
            end_frame += 1
        voice_activity[start_frame:end_frame] = True

    return voice_activity


def extract_melody(
    audio: torch.Tensor,
    sample_rate: int,
    onset_strategy: str = "librosa",
    pitch_strategy: str = "pesto",
    frame_size_millis: int = 10,
    median_filter_size: int = 5,
    min_note_duration: float = 0.1,
    max_note_duration: float = 2.0,
    offset_relative_threshold_db: float = -6.0,
    offset_absolute_threshold_db: float = -48.0,
    **kwargs,
):
    """
    Extract the melody from an audio file.
    This function takes an audio tensor and its sample rate, and extracts the melody
    by first estimating the pitch, then applying post-processing steps to improve the
    quality of the extracted melody.
    Args:
        audio (torch.Tensor): The audio signal as a PyTorch tensor.
        sample_rate (int): The sample rate of the audio signal in Hz.
        pitch_strategy (str, optional): The strategy to use for pitch estimation.
            Default is 'pesto'.
        frame_size_millis (int, optional): The size of each frame in milliseconds for
            pitch estimation. Default is 10.
    Returns:
        list: A list of tuples, each containing (pitch, start_time, duration, velocity)
              for a note in the melody.
              - pitch: MIDI note number
              - start_time: start time in seconds
              - duration: note duration in seconds
              - velocity: note velocity/volume (0.0-1.0)
    Notes:
        The function applies the following post-processing steps:
        1. Removes pitches with low confidence
        2. Applies median filtering to reduce noise
        3. Merges consecutive notes of the same pitch
    """

    timesteps, pitches, pitch_confidence = estimate_pitch(
        audio, sample_rate, frame_size_millis, pitch_strategy
    )

    # # Zero out pitches with low confidence
    # pitches[pitch_confidence < 0.05] = 0

    # Median Filtering
    median_filter_radius = median_filter_size // 2
    for i in range(median_filter_radius, len(pitches) - median_filter_radius):
        pitches[i] = torch.median(
            pitches[i - median_filter_radius : i + median_filter_radius + 1]
        )

    vad = get_vad(
        audio,
        sample_rate,
        frame_size_millis,
        strategy=onset_strategy,
        min_voice_duration_millis=min_note_duration * 1000,
        max_voice_duration_millis=max_note_duration * 1000,
        offset_relative_threshold_db=offset_relative_threshold_db,
        offset_absolute_threshold_db=offset_absolute_threshold_db,
    )

    rms = get_rms_energy(audio, sample_rate, frame_size_millis, window="rectangular")
    rms = rms / rms.max()  # Normalize RMS energy

    rms_db = torch.from_numpy(librosa.amplitude_to_db(rms.numpy(), ref=1.0))

    rms_flux = torch.roll(rms, -1) - rms
    rms_flux[-1] = 0
    rectified_rms_flux = torch.maximum(rms_flux, torch.zeros_like(rms_flux))
    rectified_rms_flux = rectified_rms_flux / rectified_rms_flux.max()

    fft_size = next_power_of_two(int(sample_rate * frame_size_millis / 1000.0))
    spectrogram = torchaudio.functional.spectrogram(
        audio,
        pad=0,
        window=torch.hann_window(fft_size),
        n_fft=fft_size,
        hop_length=int(sample_rate * frame_size_millis / 1000.0),
        win_length=fft_size,
        power=2,
        normalized=True,
    )
    spectral_flux = torch.roll(spectrogram, -1, dims=-1) - spectrogram
    spectral_flux[:, -1] = 0
    rectified_spectral_flux = torch.maximum(
        spectral_flux, torch.zeros_like(spectral_flux)
    ).mean(dim=0)
    rectified_spectral_flux = rectified_spectral_flux / rectified_spectral_flux.max()

    if kwargs.get("ax_plot"):
        ax = kwargs["ax_plot"]

        assert isinstance(ax, plt.Axes), "ax_plot must be a matplotlib Axes object"

        ax.plot(timesteps, pitches / 127, label="Pitches")
        ax.plot(timesteps, pitch_confidence, label="Confidence", alpha=0.75)
        ax.plot(timesteps, rms, label="RMS Energy", alpha=0.75)
        ax.plot(timesteps, rectified_rms_flux, label="RMS Flux", alpha=0.75)
        ax.plot(timesteps, rectified_spectral_flux, label="Spectral Flux", alpha=0.75)
        ax.plot(timesteps, vad.float(), label="VAD", alpha=0.75)

    # voice_activity = vad(audio, sample_rate, frame_size_millis)
    # # print(voice_activity.shape)
    notes = (pitches > 0) & (vad)
    note_onsets = torch.argwhere(notes.roll(-1) & ~notes).squeeze().tolist()
    note_offsets = torch.argwhere(~notes.roll(-1) & notes).squeeze().tolist()

    if kwargs.get("ax_plot"):
        ax = kwargs["ax_plot"]
        ax.vlines(
            timesteps[note_onsets],
            ymin=0,
            ymax=1,
            color="red",
            label="Note Onsets",
            alpha=0.5,
        )
        ax.vlines(
            timesteps[note_offsets],
            ymin=0,
            ymax=1,
            color="green",
            label="Note Offsets",
            alpha=0.5,
        )

    melody = []
    for onset, offset in zip(note_onsets, note_offsets):
        assert onset < offset, "Onset must be before offset"

        pitch = pitches[onset:offset].mode()[0].item()
        if pitch == 0:
            continue

        start_time = timesteps[onset].item()
        duration = timesteps[offset].item() - start_time

        if duration < min_note_duration:
            continue
        if duration > max_note_duration:
            duration = max_note_duration

        note_rms = (rms_db[onset:offset].mean().item() - offset_absolute_threshold_db) / (rms_db.max().item() - offset_absolute_threshold_db)
        rms_confidence_weight = 0.8
        velocity = 0.2 + 0.8 * (
            rms_confidence_weight * note_rms
            + (1.0 - rms_confidence_weight)
            * pitch_confidence[onset:offset].mean().item()
        )

        melody.append((pitch, start_time, duration, velocity))
    
    if kwargs.get("ax_plot"):
        ax = kwargs["ax_plot"]
        for pitch, start_time, duration, velocity in melody:
            ax.hlines(
                pitch / 127,
                start_time,
                start_time + duration,
                color="blue",
                alpha=0.5,
                label="Melody Note" if not ax.get_legend_handles_labels()[1] else "",
            )
            ax.text(
                start_time + duration / 2,
                pitch / 127 + 0.02,
                f"{pitch}",
                ha="center",
                va="bottom",
            )

    return melody


if __name__ == "__main__":
    audio, Fs = torchaudio.load(os.path.join("data", "el niño pelon.mp3"))
    audio = audio.mean(dim=0)  # convert to mono

    fig, ax = plt.subplots(figsize=(12, 6))

    extract_melody(
        audio,
        Fs,
        onset_strategy="rms_energy",
        pitch_strategy="pesto",
        frame_size_millis=10,
        ax_plot=ax,
    )

    librosa.display.waveshow(
        audio.numpy(), sr=Fs, ax=ax, alpha=0.5, label="Audio Waveform"
    )
    ax.set_xlabel("Time (s)")
    ax.legend()
    plt.show()
