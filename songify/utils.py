import symusic

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

def melody_with_harmony_to_score(
        melody: list[tuple[int, float, float, float]],
        harmony: list[tuple[list[int], float, float]]
    ):
    """
    Convert a melody and its corresponding harmony into a symusic Score.

    Each tuple in the melody should be in the format:
    (
        pitch (MIDI note number),
        start_time (seconds),
        duration (seconds),
        velocity (float in range [0, 1]),
    )

    Each tuple in the harmony should be in the format:
    (
        chord midi notes (list[int]),
        start_time (seconds),
        duration (seconds),
    )
    """
    score = melody_to_score(melody)

    for chord_notes, start_time, duration in harmony:
        for pitch in chord_notes:
            new_note = symusic.Note(
                time=start_time,
                duration=duration,
                pitch=pitch,
                velocity=60,
                ttype='second'
            )
            score.tracks[0].notes.append(new_note)

    # score.tracks[0].notes.sort(key=lambda note: note.time)
    return score



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
        ([62, 65, 69], 0.0, 2.0),
        ([62, 65, 67, 71], 2.0, 1.5),
        ([60, 64, 67, 71], 3.5, 1.0),
    ]

    harmony_score = melody_with_harmony_to_score(melody, harmony)
    print("Melody with Harmony Score:")
    print(harmony_score)

    # Save the score to a MIDI file
    harmony_score.dump_midi('output.mid')
    print("MIDI file saved as output.mid")