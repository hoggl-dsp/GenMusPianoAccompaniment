import torch
import torchaudio

import symusic
import librosa
import pesto

import numpy as np
import matplotlib.pyplot as plt

import os

import time

from songify import melody, harmonise, utils

def main():
    file = os.path.join('data', 'Capn Holt 1.mp3')

    # extract melody
    audio, sample_rate = torchaudio.load(file)
    audio = audio.mean(dim=0)  # convert to mono
    
    assert audio.dim() == 1, "Audio should be mono"

    extracted_melody = melody.extract_melody(
        audio=audio,
        sample_rate=sample_rate,
        pitch_strategy='pesto',
        frame_size_millis=50,
    )

    # generate harmony

    # Replace below with utils.melody_with_harmony_to_score when ready.
    score = utils.melody_to_score(
        melody=extracted_melody,
    )

    # Synthesize the score to audio, using symusic default piano soundfont
    piano_audio = utils.synthesise_score(score, sample_rate=sample_rate)
    
    # Mix original audio with synthesized piano audio
    expanded_audio = audio.unsqueeze(0).expand((2, -1))
    if expanded_audio.size(1) < piano_audio.size(1):
        expanded_audio = torch.cat((expanded_audio, torch.zeros((2, piano_audio.size(1) - expanded_audio.size(1)))), dim=-1)
    mixed_audio = 0.5 * expanded_audio + piano_audio

    torchaudio.save(os.path.join('output', 'melody.wav'), mixed_audio, sample_rate)

if __name__ == "__main__":
    main()