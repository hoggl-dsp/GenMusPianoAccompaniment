import torch
import torchaudio

import os
import time

from songify import melody, utils
from songify import melody_harmonizer as mh

from dataclasses import dataclass
from typing import Any, List, Tuple, Union


@dataclass
class MelodyExtractionParameters:
    onset_detection: str = "option1"  # Options: "option1", "option2", "option3"
    pitch_algorithm: str = "pesto"  # Options: "pesto", "librosa"
    frame_size: int = 512
    median_filter: int = 5
    min_note_duration: float = 0.1
    max_note_duration: float = 2.0


@dataclass
class HarmonyGenerationParameters:
    chord_melody_congruence: float = 0.5
    chord_variety: float = 0.5
    harmonic_flow: float = 0.5
    dissonance: float = 0.5
    duration_threshold: float = 0.5  # seconds


class SongifyApp:
    def __init__(
        self,
        melody_params: MelodyExtractionParameters = MelodyExtractionParameters(),
        harmony_params: HarmonyGenerationParameters = HarmonyGenerationParameters(),
    ):
        self.melody_params = melody_params
        self.harmony_params = harmony_params
        self.audio: Union[torch.Tensor, None] = None
        self.sample_rate: Union[int, None] = None

    def load_audio(self, streamlit_file):
        # file = os.path.join('data', 'Capn Holt 1.mp3')
        streamlit_file.seek(0)  
        self.audio, self.sample_rate = torchaudio.load(streamlit_file)

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


        assert self.audio is not None, "Audio must be loaded before generating music"
        assert self.sample_rate is not None, "Sample rate must be set before generating music"

        audio = self.audio.clone()
        if audio.dim() == 2:
            audio = audio.mean(dim=0)
        assert audio.dim() == 1, "Audio should be mono"

        # Noise Reduction?


        # Extract melody from audio
        extracted_melody = melody.extract_melody(
            audio=audio,
            sample_rate=self.sample_rate,
            onset_strategy=melody_params.onset_detection.lower(),
            pitch_strategy=melody_params.pitch_algorithm.lower(),
            frame_size_millis=melody_params.frame_size,
            median_filter_size=melody_params.median_filter,
            min_note_duration=melody_params.min_note_duration,
            max_note_duration=melody_params.max_note_duration,
        )

        print("Extracted Melody:", extracted_melody)

        # Generate harmony from melody
        generated_harmony = mh.harmonize(
            extracted_melody,
            congruence=harmony_params.chord_melody_congruence,
            variety=harmony_params.chord_variety,
            flow=harmony_params.harmonic_flow,
            dissonance=harmony_params.dissonance,
            duration_threshold=harmony_params.duration_threshold,
        )

        print("Generated Harmony:", generated_harmony)

        melody_score = utils.melody_to_score(extracted_melody)
        harmony_score = utils.harmony_to_score(generated_harmony)

        melody_audio = utils.synthesise_score(melody_score, sample_rate=self.sample_rate)
        harmony_audio = utils.synthesise_score(harmony_score, sample_rate=self.sample_rate)

        return melody_score, harmony_score, melody_audio, harmony_audio

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