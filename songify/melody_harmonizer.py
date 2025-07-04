import random
from dataclasses import dataclass

import music21
from preferred_chord_transitions import (IMPOSSIBLE_COMBINATIONS,
                                         chord_mappings, preferred_transitions, cadence_resolutions)

from songify import utils

chord_tuple = tuple[list[int], float, float, float]  # (notes, start, duration, velocity)

# @dataclass(frozen=True)
class MelodyData:
    """
    A data class representing the data of a melody.

    This class encapsulates the details of a melody including its notes, total
    duration, and the number of bars. The notes are represented as a list of
    tuples, with each tuple containing a pitch and its duration. The total
    duration and the number of bars are computed based on the notes provided.

    Attributes:
        notes (list of tuples): List of tuples representing the melody's notes.
            Each tuple is in the format (pitch, duration).
        duration (int): Total duration of the melody, computed from notes.
        number_of_bars (int): Total number of bars in the melody, computed from
            the duration assuming a 4/4 time signature.

    Methods:
        __post_init__: A method called after the data class initialization to
            calculate and set the duration and number of bars based on the
            provided notes.
    """

    notes: list
    duration: int
    number_of_notes: int = None  # Computed attribute

    def __init__(self, notes, durations):
        self.notes = notes
        self.durations = durations
        self.total_duration = sum(self.durations)

        self.number_of_notes = len(self.notes)

        assert self.number_of_notes == len(self.durations)

    # def __post_init__(self):
    #     object.__setattr__(self, "duration",
    #                        sum(duration for _, duration in self.notes))
    #     object.__setattr__(self, "number_of_notes", len(self.notes))


class GeneticMelodyHarmonizer:
    """
    Generates chord accompaniments for a given melody using a genetic algorithm.
    It evolves a population of chord sequences to find one that best fits the
    melody based on a fitness function.

    Attributes:
        melody_data (MelodyData): Data containing melody information.
        chords (list): Available chords for generating sequences.
        population_size (int): Size of the chord sequence population.
        mutation_rate (float): Probability of mutation in the genetic algorithm.
        fitness_evaluator (FitnessEvaluator): Instance used to assess fitness.
    """

    def __init__(
        self,
        melody_data,
        chords,
        population_size,
        mutation_rate,
        fitness_evaluator,
    ):
        """
        Initializes the generator with melody data, chords, population size,
        mutation rate, and a fitness evaluator.

        Parameters:
            melody_data (MusicData): Melody information.
            chords (list): Available chords.
            population_size (int): Size of population in the algorithm.
            mutation_rate (float): Mutation probability per chord.
            fitness_evaluator (FitnessEvaluator): Evaluator for chord fitness.
        """
        self.melody_data = melody_data
        self.chords = chords
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.fitness_evaluator = fitness_evaluator
        self._population = []

    def generate(self, generations=1000):
        """
        Generates a chord sequence that harmonizes a melody using a genetic
        algorithm.

        Parameters:
            generations (int): Number of generations for evolution.

        Returns:
            best_chord_sequence (list): Harmonization with the highest fitness
                found in the last generation.
        """
        self._population = self._initialise_population()
        for _ in range(generations):
            parents = self._select_parents()
            new_population = self._create_new_population(parents)
            self._population = new_population
        best_chord_sequence = (
            self.fitness_evaluator.get_chord_sequence_with_highest_fitness(
                self._population
            )
        )
        return best_chord_sequence

    def _initialise_population(self):
        """
        Initializes population with random chord sequences.

        Returns:
            list: List of randomly generated chord sequences.
        """
        return [
            self._generate_random_chord_sequence() for _ in range(self.population_size)
        ]

    def _generate_random_chord_sequence(self):
        """
        Generate a random chord sequence with as many chords as the numbers
        of bars in the melody.

        Returns:
            list: List of randomly generated chords.
        """
        return [
            random.choice(self.chords) for _ in range(self.melody_data.number_of_notes)
        ]

    def _select_parents(self):
        """
        Selects parent sequences for breeding based on fitness.

        Returns:
            list: Selected parent chord sequences.
        """
        fitness_values = [
            self.fitness_evaluator.evaluate(seq) for seq in self._population
        ]
        return random.choices(
            self._population, weights=fitness_values, k=self.population_size
        )

    def _create_new_population(self, parents):
        """
        Generates a new population of chord sequences from the provided parents.

        This method creates a new generation of chord sequences using crossover
        and mutation operations. For each pair of parent chord sequences,
        it generates two children. Each child is the result of a crossover
        operation between the pair of parents, followed by a potential
        mutation. The new population is formed by collecting all these
        children.

        The method ensures that the new population size is equal to the
        predefined population size of the generator. It processes parents in
        pairs, and for each pair, two children are generated.

        Parameters:
            parents (list): A list of parent chord sequences from which to
                generate the new population.

        Returns:
            list: A new population of chord sequences, generated from the
                parents.

        Note:
            This method assumes an even population size and that the number of
            parents is equal to the predefined population size.
        """
        new_population = []
        for i in range(0, self.population_size, 2):
            child1, child2 = self._crossover(
                parents[i], parents[i + 1]
            ), self._crossover(parents[i + 1], parents[i])
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)
            new_population.extend([child1, child2])
        return new_population

    def _crossover(self, parent1, parent2):
        """
        Combines two parent sequences into a new child sequence using one-point
        crossover.

        Parameters:
            parent1 (list): First parent chord sequence.
            parent2 (list): Second parent chord sequence.

        Returns:
            list: Resulting child chord sequence.
        """
        cut_index = random.randint(1, len(parent1) - 1)
        return parent1[:cut_index] + parent2[cut_index:]

    def _mutate(self, chord_sequence):
        """
        Mutates a chord in the sequence based on mutation rate.

        Parameters:
            chord_sequence (list): Chord sequence to mutate.

        Returns:
            list: Mutated chord sequence.
        """
        if random.random() < self.mutation_rate:
            mutation_index = random.randint(0, len(chord_sequence) - 1)
            chord_sequence[mutation_index] = random.choice(self.chords)
        return chord_sequence


class FitnessEvaluator:
    """
    Evaluates the fitness of a chord sequence based on various musical criteria.

    Attributes:
        melody (list): List of tuples representing notes as (pitch, duration).
        chords (dict): Dictionary of chords with their corresponding notes.
        weights (dict): Weights for different fitness evaluation functions.
        preferred_transitions (dict): Preferred chord transitions.
    """

    def __init__(self, melody_data, chord_mappings, weights, preferred_transitions):
        """
        Initialize the FitnessEvaluator with melody, chords, weights, and
        preferred transitions.

        Parameters:
            melody_data (MelodyData): Melody information.
            chord_mappings (dict): Available chords mapped to their notes.
            weights (dict): Weights for each fitness evaluation function.
            preferred_transitions (dict): Preferred chord transitions.
        """
        self.melody_data = melody_data
        self.chord_mappings = chord_mappings
        self.weights = weights
        self.preferred_transitions = preferred_transitions

    def get_chord_sequence_with_highest_fitness(self, chord_sequences):
        """
        Returns the chord sequence with the highest fitness score.

        Parameters:
            chord_sequences (list): List of chord sequences to evaluate.

        Returns:
            list: Chord sequence with the highest fitness score.
        """
        return max(chord_sequences, key=self.evaluate)

    def evaluate(self, chord_sequence):
        """
        Evaluate the fitness of a given chord sequence.

        Parameters:
            chord_sequence (list): The chord sequence to evaluate.

        Returns:
            float: The overall fitness score of the chord sequence.
        """
        return sum(
            self.weights[func] * getattr(self, f"_{func}")(chord_sequence)
            for func in self.weights
        )

    def _chord_melody_congruence(self, chord_sequence):
        """
        Calculates the congruence between the chord sequence and the melody.
        This function assesses how well each chord in the sequence aligns
        with the corresponding segment of the melody. The alignment is
        measured by checking if the notes in the melody are present in the
        chords being played at the same time, rewarding sequences where the
        melody notes fit well with the chords.

        Parameters:
            chord_sequence (list): A list of chords to be evaluated against the
                melody.

        Returns:
            float: A score representing the degree of congruence between the
                chord sequence and the melody, normalized by the melody's
                duration.
        """
        score = 0

        # print(list(enumerate(
        #         zip(chord_sequence, self.melody_data.notes, self.melody_data.durations))))

        for i, (chord, (pitch, duration)) in enumerate(
            zip(chord_sequence, zip(self.melody_data.notes, self.melody_data.durations))
        ):
            start = max(0, i - 2)
            end = min(len(self.melody_data.notes), i + 3)
            window_notes = self.melody_data.notes[start:end]
            window_pitches = [note[0] for note in window_notes]

            if any(p in self.chord_mappings[chord] for p in window_pitches):
                try:
                    score += duration
                except:
                    print(f"chord_sequence: {chord_sequence}")

        return score / self.melody_data.total_duration

    def _chord_variety(self, chord_sequence):
        """
        Evaluates the diversity of chords used in the sequence. This function
        calculates a score based on the number of unique chords present in the
        sequence compared to the total available chords. Higher variety in the
        chord sequence results in a higher score, promoting musical
        complexity and interest.

        Parameters:
            chord_sequence (list): The chord sequence to evaluate.

        Returns:
            float: A normalized score representing the variety of chords in the
                sequence relative to the total number of available chords.
        """
        unique_chords = len(set(chord_sequence))
        total_chords = len(self.chord_mappings)
        return unique_chords / total_chords

    def _harmonic_flow(self, chord_sequence):
        """
        Assesses the harmonic flow of the chord sequence by examining the
        transitions between successive chords. This function scores the
        sequence based on how frequently the chord transitions align with
        predefined preferred transitions. Smooth and musically pleasant
        transitions result in a higher score.

        Parameters:
            chord_sequence (list): The chord sequence to evaluate.

        Returns:
            float: A normalized score based on the frequency of preferred chord
                transitions in the sequence.
        """
        score = 0
        for i in range(len(chord_sequence) - 1):
            current_chord = chord_sequence[i]
            next_chord = chord_sequence[i + 1]
            if current_chord in self.preferred_transitions:
                if next_chord in self.preferred_transitions[chord_sequence[i]]:
                    score += 1
        return score / (len(chord_sequence) - 1)

    def _penalize_dissonance(self, chord_sequence):
        """
        Penalizes chord-note combinations that are considered dissonant.
        Uses the IMPOSSIBLE_COMBINATIONS dictionary to subtract points for
        any melody note in the local window that clashes with the chord.

        Returns:
            float: Penalty score (lower is worse).
        """
        penalty = 0
        for i, chord in enumerate(chord_sequence):
            if chord not in IMPOSSIBLE_COMBINATIONS:
                continue
            dissonant_pitches = IMPOSSIBLE_COMBINATIONS[chord]
            start = max(0, i - 2)
            end = min(len(self.melody_data.notes), i + 3)
            window_notes = self.melody_data.notes[start:end]
            window_pitches = [note[0] for note in window_notes]
            clashes = [p for p in window_pitches if p in dissonant_pitches]
            penalty += len(clashes)

        # Normalize: subtract from 1 so fewer penalties is better (like other fitness metrics)
        max_possible = (
            len(chord_sequence) * 5
        )  # worst case: 5 dissonant notes per chord
        return 1 - (penalty / max_possible if max_possible else 0)

    def _cadence_fitness(self, chord_sequence):
    # """
    # Evaluates whether the last two chords in the sequence form a cadence.
    # Adds a bonus score if they do.

    # Args:
    #     chord_sequence (list of str): A list of chord names (e.g., ["C", "G", "Am", "F"]).

    # Returns:
    #     int: Cadence bonus score (0 if no cadence, positive bonus if cadence found).
    # """
        if len(chord_sequence) < 2:
            return 0  # Not enough chords to evaluate cadence

        penultimate_chord = chord_sequence[-2]
        final_chord = chord_sequence[-1]

        # Check if penultimate chord resolves to final chord
        if final_chord in cadence_resolutions:
            if penultimate_chord in cadence_resolutions[final_chord]:
                return 1  # Bonus for cadence

        return 0  # No cadence found


def create_score(melody, starts, chord_sequence, chord_mappings):
    """
    Create a music21 score with a given melody and chord sequence.

    Args:
        melody (list): A list of tuples representing notes in the format
            (note_name, duration).
        chord_sequence (list): A list of chord names.

    Returns:
        music21.stream.Score: A music score containing the melody and chord
            sequence.
    """
    # Create a Score object
    score = music21.stream.Score()
    melody_part = music21.stream.Part()
    chord_part = music21.stream.Part()

    current_offset = 0
    for (note_name, duration), start, chord_name in zip(melody, starts, chord_sequence):
        melody_note = music21.note.Note(note_name)
        melody_note.seconds = duration
        melody_note.offset = start
        melody_part.append(melody_note)

        chord_notes = chord_mappings.get(chord_name, [])
        chord = music21.chord.Chord(chord_notes, quarterLength=duration)
        chord.offset = current_offset
        chord_part.append(chord)

        current_offset += duration

    # Append parts to the score
    score.append(melody_part)
    score.append(chord_part)

    return score


def midi_note_to_note_string(note_number):
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    note = note_names[note_number % 12]
    octave = (note_number // 12) - 1  # MIDI standard: C4 = 60
    return f"{note}{octave}"


def note_string_to_midi_note(note_string):
    note_names = {
        "C": 0,
        "C#": 1,
        "Db": 1,
        "D": 2,
        "D#": 3,
        "Eb": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "Gb": 6,
        "G": 7,
        "G#": 8,
        "Ab": 8,
        "A": 9,
        "A#": 10,
        "Bb": 10,
        "B": 11,
    }

    for name in sorted(note_names.keys(), key=len, reverse=True):
        if note_string.startswith(name):
            try:
                octave = int(note_string[len(name) :])
            except:
                octave = 4

            return (octave + 1) * 12 + note_names[name]
    raise ValueError(f"Invalid note string: {note_string}")


def chord_strings_to_midi_chords(chords):
    # Chord string to note strings list
    chords = [chord_mappings[chord] for chord in chords]

    # Note string list to MIDI integer note list

    chords = [
        [note_string_to_midi_note(string_note) for string_note in notes]
        for notes in chords
    ]

    return chords


def invert_chords_below_melody(midi_chords, melody_midi_notes):
    """
    Inverts chord notes so they fall below the corresponding melody note in MIDI pitch.

    Args:
        midi_chords (list of list of ints): Chord notes in MIDI numbers.
        melody_midi_notes (list of ints): Melody notes in MIDI numbers.

    Returns:
        list of list of ints: Inverted chord MIDI note lists.
    """
    inverted_chords = []

    for chord, melody_note in zip(midi_chords, melody_midi_notes):
        inverted_chord = []
        for note in chord:
            # If a chord note is higher than the melody note, drop it by an octave
            while note >= melody_note:
                note -= 12
            inverted_chord.append(note)
        inverted_chords.append(inverted_chord)

    return inverted_chords


def filter_chords_by_duration(harmony, duration_threshold=0.1):
    """
    Remove chords whose corresponding melody notes are too short.

    Args:
        harmony (iterable): List of (chord_notes, start, duration, velocity) tuples.
        duration_threshold (float): Chords are removed if duration is below this.

    Returns:
        list: Filtered harmony list.
    """
    filtered_harmony = []
    for chord_notes, start, duration, velocity in harmony:
        if duration < duration_threshold:
            filtered_harmony.append(([], start, duration, velocity))
        else:
            filtered_harmony.append((chord_notes, start, duration, velocity))
    return filtered_harmony

def fill_arpeggio_between_chords(
    harmony_sequence: list[chord_tuple],
    k: int,
    direction: str = "ascending"
) -> list[chord_tuple]:
    filled = harmony_sequence.copy()
    empty_indices = [i for i, (chord, *_rest) in enumerate(filled) if not chord]

    if k > len(empty_indices):
        k = len(empty_indices)
    if k == 0:
        return filled

    # Choose k empty slots to fill (preserve timing order)
    selected_indices = sorted(random.sample(empty_indices, k))

    # Process each
    for idx in selected_indices:
        # Find previous and next non-empty chords
        prev_chord = next_chord = []
        for i in range(idx - 1, -1, -1):
            if filled[i][0]:
                prev_chord = filled[i][0]
                break
        for i in range(idx + 1, len(filled)):
            if filled[i][0]:
                next_chord = filled[i][0]
                break

        # Merge pitch content
        chord_pool = sorted(set(prev_chord + next_chord))

        # If no real chord context, skip
        if not chord_pool:
            continue

        # Create arpeggio line
        if direction == "ascending":
            arpeggio = chord_pool
        elif direction == "descending":
            arpeggio = chord_pool[::-1]
        elif direction == "random":
            arpeggio = chord_pool[:]
            random.shuffle(arpeggio)
        else:
            raise ValueError(f"Unknown direction: {direction}")

        # Pick arpeggio note using round-robin
        note = [arpeggio[idx % len(arpeggio)]]

        # Preserve original timing
        _, start, dur, vel = filled[idx]
        filled[idx] = (note, start, dur, vel * 0.5)

    return filled

def harmonize(data, congruence, variety, flow, dissonance, cadence, duration_threshold, population_size,generations):
    pitch = [midi_note_to_note_string(note[0]) for note in data]
    starts = [note[1] for note in data]
    durations = [note[2] for note in data]
    velocities = [note[3] for note in data]

    weights = {
        "chord_melody_congruence": congruence,
        "chord_variety": variety,
        "harmonic_flow": flow,
        "penalize_dissonance": dissonance,
        "cadence_fitness": cadence,
    }

    # Instantiate objects for generating harmonization

    melody_data = MelodyData(pitch, durations)
    fitness_evaluator = FitnessEvaluator(
        melody_data=melody_data,
        weights=weights,
        chord_mappings=chord_mappings,
        preferred_transitions=preferred_transitions,
    )
    harmonizer = GeneticMelodyHarmonizer(
        melody_data=melody_data,
        chords=list(chord_mappings.keys()),
        population_size=population_size, 
        mutation_rate=0.05,
        fitness_evaluator=fitness_evaluator,
    )

    # Generate chords with genetic algorithm
    harmony = harmonizer.generate(generations)
    harmony = fill_arpeggio_between_chords(harmony,k=10,direction = "random")
    harmony = chord_strings_to_midi_chords(harmony)
    melody_midi_notes = [note_string_to_midi_note(note) for note in pitch]
    inverted_harmony = invert_chords_below_melody(harmony, melody_midi_notes)
    harmony = zip(inverted_harmony, starts, durations, velocities)
    harmony = filter_chords_by_duration(harmony, duration_threshold)

    return harmony


if __name__ == "__main__":
    main()
