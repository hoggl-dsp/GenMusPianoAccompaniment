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
    def __init__(self, melody_params: MelodyExtractionParameters = MelodyExtractionParameters(),
                 harmony_params: HarmonyGenerationParameters = HarmonyGenerationParameters()):
        self.melody_params = melody_params
        self.harmony_params = harmony_params
        self.audio_path = None

    def load_audio(self, audio_path: str):
        pass

    def _set_melody_params(self, melody_params: MelodyExtractionParameters):
        self.melody_params = melody_params

    def _set_harmony_params(self, harmony_params: HarmonyGenerationParameters):
        self.harmony_params = harmony_params

    # Extract melody from audio file, and return annotated melody (plot)
    def extract_melody(self, audio, sample_rate) -> List[Tuple[Any, Any, Any]]:
        pass

    def generate(self, melody_params: MelodyExtractionParameters, harmony_params: HarmonyGenerationParameters):
        pass

    def get_generated_wav_file(self) -> str:
        """
        Returns the path to the generated WAV file.
        """
        return "path/to/generated.wav"
    
    def get_generated_midi_file(self) -> str:
        return "path/to/generated.mid"
