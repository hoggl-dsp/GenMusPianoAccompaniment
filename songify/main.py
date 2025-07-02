import torch
import torchaudio

import os
import time

from songify import melody, utils
from songify import melody_harmonizer as mh

from dataclasses import dataclass
from typing import Any, List, Tuple


@dataclass
class MelodyExtractionParameters:
    onset_detection: str = "option1"  # Options: "option1", "option2", "option3"
    frame_size: float = 512
    median_filter: float = 65.0
    min_note_duration: float = 0.1
    max_note_duration: float = 2.0


@dataclass
class HarmonyGenerationParameters:
    chord_melody_congruence: float = 0.5
    chord_variety: float = 0.5
    harmonic_flow: float = 0.5
    functional_harmony: float = 0.5


class SongifyApp:
    def __init__(
        self,
        melody_params: MelodyExtractionParameters = MelodyExtractionParameters(),
        harmony_params: HarmonyGenerationParameters = HarmonyGenerationParameters(),
    ):
        self.melody_params = melody_params
        self.harmony_params = harmony_params
        self.audio = None
        self.sample_rate = None

    def load_audio(self, streamlit_file):
        # file = os.path.join('data', 'Capn Holt 1.mp3')
        streamlit_file.seek(0)  
        self.audio, self.sample_rate = torchaudio.load(streamlit_file)
        # print(f"Loading audio from {audio_path}")

    # Extract melody from audio file, and return annotated melody (plot)
    def extract_melody(self) -> List[Tuple[Any, Any, Any]]:
        print("Extracting melody with parameters:", self.melody_params)
        pass

    def generate(
        self,
        melody_params: MelodyExtractionParameters,
        harmony_params: HarmonyGenerationParameters,
    ):
        print("Generating music with parameters:")
        print("Melody Parameters:", melody_params)
        print("Harmony Parameters:", harmony_params)


        pass

    def get_generated_wav_file(self) -> str:
        """
        Returns the path to the generated WAV file.
        """
        return "path/to/generated.wav"

    def get_generated_midi_file(self) -> str:
        return "path/to/generated.mid"


def main(merge: bool = True):
    # file = os.path.join('data', 'Capn Holt 2.mp3')
    # file = os.path.join('data', 'can i pet that dog.mp3')
    file = os.path.join('data', 'drop my croissant.mp3')

    # extract melody
    original_audio, sample_rate = torchaudio.load(file)
    audio = original_audio.mean(dim=0)  # convert to mono
    
    assert audio.dim() == 1, "Audio should be mono"

    audio = torchaudio.functional.vad(audio, sample_rate=sample_rate)
    if audio.size(0) < audio.size(0):
        # If VAD reduces the length, pad the audio to maintain size
        padding = torch.zeros(audio.size(0) - audio.size(0))
        audio = torch.cat((audio, padding), dim=0)
    else:
        # If VAD does not reduce the length, ensure the size matches
        audio = audio[:audio.size(0)]
    
    assert audio.size() == audio.size(), "Denoised audio should have the same size as original audio"

    extracted_melody = melody.extract_melody(
        audio=audio,
        sample_rate=sample_rate,
        pitch_strategy='pesto',
        frame_size_millis=10,
    )

    melody_score = utils.melody_to_score(
        melody=extracted_melody,
    )

    harmony = mh.harmonize(extracted_melody)
    harmony_score = utils.harmony_to_score(harmony)


    harmony_score.dump_midi(os.path.join('output', 'harmony_score.mid'))

    if merge:
        score = utils.merge_scores([melody_score, harmony_score])
    else:
        score = harmony_score

    score.dump_midi(os.path.join('output', 'merged_score.mid'))

    # Synthesize the score to audio, using symusic default piano soundfont
    piano_audio = utils.synthesise_score(score, sample_rate=sample_rate)
    
    # Mix original audio with synthesized piano audio
    mixed_audio = utils.mix_audio(
        original_audio=original_audio,
        synthesized_audio=piano_audio,
        blend=0.5,  # Adjust blend as needed
        stereo=True
    )

    torchaudio.save(os.path.join('output', 'output.wav'), mixed_audio, sample_rate)

if __name__ == "__main__":
    main()