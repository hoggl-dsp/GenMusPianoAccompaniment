import re

from music21 import pitch, scale

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
    "Baug": ["B", "D#", "F##"],
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
    match = re.match(r"^([A-G][#b]?)(m|dim|aug)?$", chord_name)
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
                preferred_transitions[chord_name] = [
                    c for c in ["C", "Em", "G"] if c in chord_mappings
                ]
            else:
                preferred_transitions[chord_name] = [
                    c for c in ["Am", "C", "F"] if c in chord_mappings
                ]
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
                scale_degrees[0] + "m",
                scale_degrees[1] + "dim",
                scale_degrees[2],
                scale_degrees[3] + "m",
                scale_degrees[4] + "m",
                scale_degrees[5],
                scale_degrees[6],
            ]
        else:
            degree_chords = [
                scale_degrees[0],
                scale_degrees[1] + "m",
                scale_degrees[2] + "m",
                scale_degrees[3],
                scale_degrees[4],
                scale_degrees[5] + "m",
                scale_degrees[6] + "dim",
            ]

        # Filter and score transitions
        candidates = [
            ch for ch in degree_chords if ch in chord_mappings and ch != chord_name
        ]
        ranked = sorted(
            candidates,
            key=lambda c: get_chord_similarity(chord_name, c, chord_mappings),
            reverse=True,
        )

        # Only include chords that share at least 2 notes (strong harmonic connection)
        filtered = [
            c
            for c in ranked
            if get_chord_similarity(chord_name, c, chord_mappings) >= 2
        ]
        preferred_transitions[chord_name] = filtered[:3]  # take top 3 best matches

    return preferred_transitions


# Generate and print
preferred_transitions = build_preferred_transitions(chord_mappings)

# for chord, transitions in preferred_transitions.items():
#     print(f"{chord}: {transitions}")

IMPOSSIBLE_COMBINATIONS = {
    # Major chords
    "C": ["F#", "A#", "D#"],
    "C#": ["G", "B", "E"],
    "D": ["G#", "C", "F"],
    "D#": ["A", "C#", "F#"],
    "E": ["A#", "D", "G"],
    "F": ["B", "D#", "G#"],
    "F#": ["C", "E", "A"],
    "G": ["C#", "F", "A#"],
    "G#": ["D", "F#", "B"],
    "A": ["D#", "G", "C"],
    "A#": ["E", "G#", "C#"],
    "B": ["F", "A", "D"],
    # Minor chords
    "Cm": ["D", "G#", "B"],
    "C#m": ["D#", "A", "C"],
    "Dm": ["E", "A#", "C#"],
    "D#m": ["F", "B", "D"],
    "Em": ["F#", "C", "D#"],
    "Fm": ["G", "C#", "E"],
    "F#m": ["G#", "D", "F"],
    "Gm": ["A", "D#", "F#"],
    "G#m": ["B", "E", "G"],
    "Am": ["C", "F", "G#"],
    "A#m": ["C#", "F#", "A"],
    "Bm": ["D", "G", "A#"],
    # Diminished chords
    "Cdim": ["D", "F", "A"],
    "C#dim": ["D#", "F#", "A#"],
    "Ddim": ["E", "G", "B"],
    "D#dim": ["F", "G#", "C"],
    "Edim": ["F#", "A", "C#"],
    "Fdim": ["G", "A#", "D"],
    "F#dim": ["G#", "B", "D#"],
    "Gdim": ["A", "C", "E"],
    "G#dim": ["A#", "C#", "F"],
    "Adim": ["B", "D", "F#"],
    "A#dim": ["C", "D#", "G"],
    "Bdim": ["C#", "E", "G#"],
    # Augmented chords
    "Caug": ["D", "F", "G"],
    "C#aug": ["D#", "F#", "G#"],
    "Daug": ["E", "G", "A"],
    "D#aug": ["F", "G#", "A#"],
    "Eaug": ["F#", "A", "B"],
    "Faug": ["G", "A#", "C"],
    "F#aug": ["G#", "B", "C#"],
    "Gaug": ["A", "C", "D"],
    "G#aug": ["A#", "C#", "D#"],
    "Aaug": ["B", "D", "E"],
    "A#aug": ["C", "D#", "F"],
    "Baug": ["C#", "E", "F#"],
}
cadence_resolutions = {
    "C": ['G', 'F', 'Am'],
    "C#": ['G#', 'F#', 'A#m'],
    "D":   ['A', 'G', 'Bm'],
    "D#":  ['A#', 'G#', 'Cm'],
    "E":   ['B', 'A', 'C#m'],
    "F":   ['C', 'A#', 'Dm'],
    "F#":  ['C#', 'B', 'D#m'],
    "G":   ['D', 'C', 'Em'],
    "G#":  ['D#', 'C#', 'Fm'],
    "A":   ['E', 'D', 'F#m'],
    "A#":  ['F', 'D#', 'Gm'],
    "B":   ['F#', 'E', 'G#m'],

    "Cm":  ['Gm', 'Fm'],
    "C#m": ['G#m', 'F#m'],
    "Dm":  ['Am', 'Gm'],
    "D#m": ['A#m', 'G#m'],
    "Em":  ['Bm', 'Am'],
    "Fm":  ['Cm', 'A#m'],
    "F#m": ['C#m', 'Bm'],
    "Gm":  ['Dm', 'Cm'],
    "G#m": ['D#m', 'C#m'],
    "Am":  ['Em', 'Dm'],
    "A#m": ['Fm', 'D#m'],
    "Bm":  ['F#m', 'Em']
}