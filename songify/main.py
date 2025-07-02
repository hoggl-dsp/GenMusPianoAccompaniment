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
from songify import melody_harmonizer as mh

def main():
    # file = os.path.join('data', 'Capn Holt 1.mp3')
    file = os.path.join('data', 'can i pet that dog.mp3')

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

    harmony = mh.harmonize(extracted_melody)
    harmony_score = utils.harmony_to_score(harmony)
    harmony_audio = utils.synthesise_score(harmony_score)
    
    harmony_score.dump_midi(os.path.join('output', 'harmony_score.mid'))
    
    final_mix = utils.mix_audio(audio, harmony_audio)
    
    torchaudio.save(os.path.join('output', 'mix.wav'), final_mix, sample_rate)

if __name__ == "__main__":
    main()