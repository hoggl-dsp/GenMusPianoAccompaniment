import symusic
import symusic.types

import torch

def melody_to_score(melody: list[tuple[int, float, float, float]]):
    """
    Convert a melody represented as a list of tuples into a symusic Score.
    
    Each tuple in the melody should be in the format:
    (
        pitch (MIDI note number),
        start_time (seconds),
        duration (seconds),
        velocity (float in range [0, 1]),
    )
    """
    track = symusic.Track("Melody", ttype='second')
    
    for pitch, start_time, duration, velocity in melody:
        new_note = symusic.Note(
            time=start_time,
            duration=duration,
            pitch=pitch,
            velocity=int(127 * velocity),
            ttype='second'
        )
        track.notes.append(new_note)
    
    score = symusic.Score(1000, ttype='second')
    score.tracks.append(track)
    
    return score

def harmony_to_score(chords: list[tuple[list[int], float, float, float]]):
    """
    Convert a list of chords into a symusic Score.

    Each tuple in the chords should be in the format:
    (
        chord midi notes (list[int]),
        start_time (seconds),
        duration (seconds),
        velocity (0 to 1)
    )
    """
    track = symusic.Track("Chords", ttype='second')
    
    for chord_notes, start_time, duration, velocity in chords:
        for pitch in chord_notes:
            new_note = symusic.Note(
                time=start_time,
                duration=duration,
                pitch=pitch,
                velocity=int(velocity * 127),
                ttype='second'
            )
            track.notes.append(new_note)
    
    score = symusic.Score(1000, ttype='second')
    score.tracks.append(track)
    
    return score

def merge_scores(
    scores: list[symusic.types.Score],
):
    """
    Merge multiple symusic Scores into a single Score, by combining all notes into a single track.
    """
    merged_score = symusic.Score(1000, ttype='second')
    merged_track = symusic.Track("Merged", ttype='second')

    for score in scores:
        for track in score.tracks:
            merged_track.notes.extend(track.notes)
    
    # Sort notes by time
    merged_track.notes.sort(key=lambda note: note.time)
    
    merged_score.tracks.append(merged_track)
    
    return merged_score

def synthesise_score(score: symusic.types.Score, sample_rate: int = 44100) -> torch.Tensor:
    """
    Synthesize the given symusic Score into audio (as a torch Tensor).
    """
    return torch.from_numpy(symusic.Synthesizer(sample_rate=sample_rate).render(score=score, stereo=True))

def mix_audio(
    original_audio: torch.Tensor,
    synthesized_audio: torch.Tensor,
    blend: float = 0.5,
    stereo: bool = True
) -> torch.Tensor:
    """
    Mix the original audio with the synthesized audio.
    
    The blend parameter controls the mix ratio:
    - 0.0 means only original audio
    - 1.0 means only synthesized audio
    """
    if original_audio.dim() > 2 or synthesized_audio.dim() > 2:
        raise ValueError("Audio tensors must be 1D or 2D (mono or stereo).")
    
    # Convert to stereo or mono
    if stereo:
        if original_audio.dim() == 1:
            original_audio = original_audio.unsqueeze(0)
        if synthesized_audio.dim() == 1:
            synthesized_audio = synthesized_audio.unsqueeze(0)
        
        if original_audio.size(0) < 2:
            original_audio = original_audio.expand((2, -1))
        if synthesized_audio.size(0) < 2:
            synthesized_audio = synthesized_audio.expand((2, -1))
    else:
        if original_audio.dim() == 2:
            original_audio = original_audio.mean(dim=0, keepdim=True)
        if synthesized_audio.dim() == 2:
            synthesized_audio = synthesized_audio.mean(dim=0, keepdim=True)
    
    # Ensure both audio tensors have the same length
    # Either truncate or pad synthesized audio to match original audio length
    if original_audio.size(1) < synthesized_audio.size(1):
        original_audio = torch.cat((original_audio, torch.zeros((2, synthesized_audio.size(1) - original_audio.size(1)))), dim=-1)
    elif original_audio.size(1) > synthesized_audio.size(1):
        synthesized_audio = torch.cat((synthesized_audio, torch.zeros((2, original_audio.size(1) - synthesized_audio.size(1)))), dim=-1)
    mixed_audio = original_audio + synthesized_audio

    return mixed_audio * blend + (1.0 - blend) * original_audio

if __name__ == '__main__':
    # Example usage
    melody = [
        (74, 0.0, 0.5, 1.0),
        (76, 0.5, 0.5, 0.8),
        (77, 1.0, 0.5, 0.7),
        (79, 1.5, 0.5, 0.8),
        (76, 2.0, 1.0, 1.0),
        (72, 3.0, 0.5, 0.9),
        (74, 3.5, 1.0, 1.0),
    ]
    
    score = melody_to_score(melody)
    print("Melody Score:")
    print(score)

    harmony = [
        ([62, 65, 69], 0.0, 2.0, 0.7),
        ([62, 65, 67, 71], 2.0, 1.5, 0.5),
        ([60, 64, 67, 71], 3.5, 1.0, 1.0),
    ]

    harmony_score = chords_to_score(harmony)
    print("Harmony Score:")
    print(harmony_score)

    # Save the score to a MIDI file
    harmony_score.dump_midi('output.mid')
    print("MIDI file saved as output.mid")