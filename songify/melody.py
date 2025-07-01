import torch
import torchaudio

import pesto
import librosa

import matplotlib.pyplot as plt
import os

from songify import utils

def estimate_pitch(
    audio: torch.Tensor,
    sample_rate: int,
    frame_size_millis: int = 10,
    pitch_strategy: str = 'pesto',
):
    if pitch_strategy == 'pesto':
        timesteps, pitches, confidence, _ = pesto.predict(audio, sample_rate, step_size=frame_size_millis)
    elif pitch_strategy == 'librosa':
        frame_samples = int(sample_rate * frame_size_millis / 1000.0)
        pitches, _, confidence = librosa.pyin(
            audio.numpy(),
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sample_rate,
            frame_length=frame_samples,
            hop_length=frame_samples,
            fill_na=False,
        )
        timesteps = torch.arange(len(pitches)) * (frame_size_millis / 1000.0)
        pitches = torch.from_numpy(pitches)
        confidence = torch.from_numpy(confidence)
    else:
        raise NotImplementedError(f"Pitch strategy '{pitch_strategy}' is not implemented.")
    
    pitches = torch.round(69 + 12 * torch.log2(pitches / 440.0)).int() 

    return timesteps, pitches, confidence

def vad(
    audio: torch.Tensor, 
    sample_rate: int,
    frame_size_millis: int = 10,
    strategy: str = 'default'
):
    """
    Voice Activity Detection (VAD) to filter out silent parts of the audio.
    
    Args:
        audio (torch.Tensor): Audio signal.
        sample_rate (int): Sample rate of the audio.
        frame_size_millis (int): Size of each frame in milliseconds.
        threshold (float): Energy threshold for VAD.
        strategy (str): Strategy for VAD, currently only 'default' is implemented.
    
    Returns:
        torch.Tensor: Filtered audio signal with silent parts removed.
    """
    if strategy == 'default':
        denoised_audio = torchaudio.functional.vad(audio, sample_rate)
        frame_size_samples = int(sample_rate * frame_size_millis / 1000.0)
        squared_audio = denoised_audio ** 2
        windowed_power = torch.conv1d(
            squared_audio.view((1, 1, -1)),
            torch.ones((1, 1, frame_size_samples)) / frame_size_samples,
            stride=frame_size_samples,padding=frame_size_samples // 2
        ).view(-1)
        rms = torch.sqrt(windowed_power)
        return rms
    # elif strategy == 'silero':
    #     model = load_silero_vad()

    #     # model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True)
    #     # (get_speech_timestamps,_, read_audio, *_) = utils
    #     target_sample_rate = 16000
    #     wav = read_audio('en_example.wav', sampling_rate=target_sample_rate)
    #     # get speech timestamps from full audio file
    #     speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=target_sample_rate)
    else:
        raise NotImplementedError(f"VAD strategy '{strategy}' is not implemented.")


def extract_melody(
    audio: torch.Tensor,
    sample_rate: int,
    pitch_strategy: str = 'pesto',
    frame_size_millis: int = 10,
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
    timesteps, pitches, pitch_confidence = estimate_pitch(audio, sample_rate, frame_size_millis, pitch_strategy)

    timesteps = timesteps / 1000.0 # Convert to seconds

    # Zero out pitches with low confidence
    pitches[pitch_confidence < 0.5] = 0

    # Median Filtering
    median_filter_size = 3
    median_filter_radius = median_filter_size // 2
    for i in range(median_filter_radius, len(pitches) - median_filter_radius):
        pitches[i] = torch.median(pitches[i - median_filter_radius: i + median_filter_radius + 1])

    

    # voice_activity = vad(audio, sample_rate, frame_size_millis)
    # # print(voice_activity.shape)

    # fig, axs = plt.subplots(1, 1, figsize=(12, 6))

    # librosa.display.waveshow(audio.numpy(), sr=sample_rate, ax=axs, alpha=0.5)
    # axs.set_title('Audio with Pitch and Voice Activity')
    # axs.set_xlabel('Time (s)')

    # axs.plot(timesteps.numpy() / 1000.0, pitch_confidence.numpy(), label='Pitch Confidence', color='orange')
    # # axs.plot(timesteps.numpy(), voice_activity.numpy(), label='Voice Activity', color='green')

    # axs.plot(timesteps.numpy() / 1000.0, pitches.numpy() / 127.0, label='Pitch (MIDI)', color='blue')

    # # plt.show()

    # start_times = (timesteps / 1000.0).tolist()
    # durations = (torch.ones_like(timesteps) * 0.99 * (frame_size_millis / 1000.0)).tolist()
    # pitches = pitches.tolist()
    # velocities = pitch_confidence.tolist()

    frame_size_secs = frame_size_millis / 1000.0

    melody = []
    for i, (timestep, pitch, confidence) in enumerate(zip(timesteps, pitches, pitch_confidence)):
        if pitch == 0:
            continue

        if len(melody) > 0 and pitch == melody[-1][0]:
            last_pitch, last_start, duration, last_vel = melody[-1]

            duration += frame_size_secs
            melody.pop()
            melody.append((last_pitch, last_start, duration, last_vel))
            continue

        start_time = timestep.item()
        duration = frame_size_secs
        velocity = confidence.item()
        melody.append((pitch.item(), start_time, duration, velocity))

    return melody


    
if __name__ == '__main__':
    audio, Fs = torchaudio.load(os.path.join('data', 'Capn Holt 1.mp3'))
    audio = audio.mean(dim=0)  # convert to mono

    extract_melody(audio, Fs, pitch_strategy='pesto', frame_size_millis=50)