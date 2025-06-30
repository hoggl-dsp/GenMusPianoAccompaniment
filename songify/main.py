import torch
import torchaudio

import symusic
import librosa
import pesto

import numpy as np
import matplotlib.pyplot as plt

import os

import time

from songify import melody, harmonise

def main():
    file = os.path.join('data', 'Capn Holt 1.mp3')

    # extract melody
    audio, sample_rate = torchaudio.load(file)
    audio = audio.mean(dim=0)  # convert to mono
    
    assert audio.dim() == 1, "Audio should be mono"

    start = time.time()
    timesteps, pitch, confidence, activations = pesto.predict(audio, sample_rate)
    end = time.time()
    print(f"Melody extraction took {end - start:.3f} seconds")

    print(timesteps)
    print(pitch)
    print(confidence)
    print(activations)

    plt.imshow(activations.T, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar()
    plt.xlabel('Time (frames)')
    plt.show()

    # generate harmony

if __name__ == "__main__":
    main()