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
    timesteps, pitches, pitch_confidence = estimate_pitch(audio, sample_rate, frame_size_millis, pitch_strategy)

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

    start_times = (timesteps / 1000.0).tolist()
    durations = (torch.ones_like(timesteps) * 0.99 * (frame_size_millis / 1000.0)).tolist()
    pitches = pitches.tolist()
    velocities = pitch_confidence.tolist()

    return list(zip(pitches, start_times, durations, velocities))


    
if __name__ == '__main__':
    audio, Fs = torchaudio.load(os.path.join('data', 'Capn Holt 1.mp3'))
    audio = audio.mean(dim=0)  # convert to mono

    extract_melody(audio, Fs, pitch_strategy='pesto', frame_size_millis=50)