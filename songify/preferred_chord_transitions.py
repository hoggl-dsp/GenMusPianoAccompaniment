from music21 import scale, pitch
import re

chord_mappings = {
    # major chords
    "C": ["C", "E", "G"],
    "C#": ["C#", "E#", "G#"],
    "D": ["D", "F#", "A"],
    "D#": ["D#", "G", "A#"],
    "E": ["E", "G#", "B"],
    "F": ["F", "A", "C"],
    "F#": ["F#", "A#", "C#"],
    "G": ["G", "B", "D"],
    "G#": ["G#", "C", "D#"],
    "A": ["A", "C#", "E"],
    "A#": ["A#", "D", "F"],
    "B": ["B", "D#", "F#"],

    # minor chords
    "Cm": ["C", "Eb", "G"],
    "C#m": ["C#", "E", "G#"],
    "Dm": ["D", "F", "A"],
    "D#m": ["D#", "F#", "A#"],
    "Em": ["E", "G", "B"],
    "Fm": ["F", "Ab", "C"],
    "F#m": ["F#", "A", "C#"],
    "Gm": ["G", "Bb", "D"],
    "G#m": ["G#", "B", "D#"],
    "Am": ["A", "C", "E"],
    "A#m": ["A#", "C#", "F"],
    "Bm": ["B", "D", "F#"],

    # diminished chords
    "Cdim": ["C", "Eb", "Gb"],
    "C#dim": ["C#", "E", "G"],
    "Ddim": ["D", "F", "Ab"],
    "D#dim": ["D#", "F#", "A"],
    "Edim": ["E", "G", "Bb"],
    "Fdim": ["F", "Ab", "C"],
    "F#dim": ["F#", "A", "C"],
    "Gdim": ["G", "Bb", "Db"],
    "G#dim": ["G#", "B", "D"],
    "Adim": ["A", "C", "Eb"],
    "A#dim": ["A#", "C#", "E"],
    "Bdim": ["B", "D", "F"],

    # augmented chords
    "Caug": ["C", "E", "G#"],
    "C#aug": ["C#", "E#", "G##"],
    "Daug": ["D", "F#", "A#"],
    "D#aug": ["D#", "G", "B"],
    "Eaug": ["E", "G#", "B#"],
    "Faug": ["F", "A", "C#"],
    "F#aug": ["F#", "A#", "C##"],
    "Gaug": ["G", "B", "D#"],
    "G#aug": ["G#", "C", "E"],
    "Aaug": ["A", "C#", "E#"],
    "A#aug": ["A#", "D", "F##"],
    "Baug": ["B", "D#", "F##"]
}


def get_chord_similarity(chord1, chord2, chord_mappings):
    """
    Computes how many notes two chords have in common.
    """
    notes1 = set(chord_mappings[chord1])
    notes2 = set(chord_mappings[chord2])
    return len(notes1.intersection(notes2))


def extract_root(chord_name):
    """
    Extracts the root note from a chord name that may end with m, dim, or aug.
    """
    match = re.match(r'^([A-G][#b]?)(m|dim|aug)?$', chord_name)
    if not match:
        raise ValueError(f"Invalid chord name: {chord_name}")
    return match.group(1)


def build_preferred_transitions(chord_mappings):
    preferred_transitions = {}

    for chord_name in chord_mappings:
        try:
            root = extract_root(chord_name)
            root = pitch.Pitch(root).name
        except Exception as e:
            print(f"Skipping {chord_name}: {e}")
            continue

        is_minor = chord_name.endswith("m") and not chord_name.endswith("dim")
        is_diminished = chord_name.endswith("dim")

        if is_diminished:
            if root in ["B", "D#", "F#"]:
                preferred_transitions[chord_name] = [c for c in ["C", "Em", "G"] if c in chord_mappings]
            else:
                preferred_transitions[chord_name] = [c for c in ["Am", "C", "F"] if c in chord_mappings]
            continue

        scale_type = scale.MinorScale if is_minor else scale.MajorScale
        try:
            tonal_scale = scale_type(root)
        except Exception as e:
            print(f"Failed to create scale for {chord_name}: {e}")
            continue

        scale_degrees = [tonal_scale.pitchFromDegree(i).name for i in range(1, 8)]

        # Determine diatonic chords based on mode
        if is_minor:
            degree_chords = [
                scale_degrees[0] + "m", scale_degrees[1] + "dim", scale_degrees[2],
                scale_degrees[3] + "m", scale_degrees[4] + "m",
                scale_degrees[5], scale_degrees[6]
            ]
        else:
            degree_chords = [
                scale_degrees[0], scale_degrees[1] + "m", scale_degrees[2] + "m",
                scale_degrees[3], scale_degrees[4],
                scale_degrees[5] + "m", scale_degrees[6] + "dim"
            ]

        # Filter and score transitions
        candidates = [ch for ch in degree_chords if ch in chord_mappings and ch != chord_name]
        ranked = sorted(
            candidates,
            key=lambda c: get_chord_similarity(chord_name, c, chord_mappings),
            reverse=True,
        )

        # Only include chords that share at least 2 notes (strong harmonic connection)
        filtered = [c for c in ranked if get_chord_similarity(chord_name, c, chord_mappings) >= 2]
        preferred_transitions[chord_name] = filtered[:3]  # take top 3 best matches

    return preferred_transitions


# Generate and print
preferred_transitions = build_preferred_transitions(chord_mappings)

# for chord, transitions in preferred_transitions.items():
#     print(f"{chord}: {transitions}")
