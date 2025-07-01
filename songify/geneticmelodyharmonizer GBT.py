import random
from dataclasses import dataclass
import music21


@dataclass(frozen=True)
class MelodyData:
    notes: list
    duration: int = None
    number_of_bars: int = None

    def __post_init__(self):
        object.__setattr__(
            self, "duration", sum(duration for _, duration in self.notes)
        )
        object.__setattr__(self, "number_of_bars", self.duration // 4)


class GeneticMelodyHarmonizer:
    def __init__(
        self,
        melody_data,
        chords,
        population_size,
        mutation_rate,
        fitness_evaluator,
    ):
        self.melody_data = melody_data
        self.chords = chords
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.fitness_evaluator = fitness_evaluator
        self._population = []

    def generate(self, generations=1000):
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
        return [
            self._generate_random_chord_sequence()
            for _ in range(self.population_size)
        ]

    def _generate_random_chord_sequence(self):
        return [
            random.choice(self.chords)
            for _ in range(len(self.melody_data.notes))
        ]

    def _select_parents(self):
        fitness_values = [
            self.fitness_evaluator.evaluate(seq) for seq in self._population
        ]
        return random.choices(
            self._population, weights=fitness_values, k=self.population_size
        )

    def _create_new_population(self, parents):
        new_population = []
        for i in range(0, self.population_size, 2):
            child1 = self._mutate(self._crossover(parents[i], parents[i + 1]))
            child2 = self._mutate(self._crossover(parents[i + 1], parents[i]))
            new_population.extend([child1, child2])
        return new_population

    def _crossover(self, parent1, parent2):
        cut_index = random.randint(1, len(parent1) - 1)
        return parent1[:cut_index] + parent2[cut_index:]

    def _mutate(self, chord_sequence):
        if random.random() < self.mutation_rate:
            mutation_index = random.randint(0, len(chord_sequence) - 1)
            chord_sequence[mutation_index] = random.choice(self.chords)
        return chord_sequence


class FitnessEvaluator:
    def __init__(
        self, melody_data, chord_mappings, weights, preferred_transitions
    ):
        self.melody_data = melody_data
        self.chord_mappings = chord_mappings
        self.weights = weights
        self.preferred_transitions = preferred_transitions

    def get_chord_sequence_with_highest_fitness(self, chord_sequences):
        return max(chord_sequences, key=self.evaluate)

    def evaluate(self, chord_sequence):
        return sum(
            self.weights[func] * getattr(self, f"_{func}")(chord_sequence)
            for func in self.weights
        )

    def _chord_melody_congruence(self, chord_sequence):
        score = 0
        for (pitch, duration), chord in zip(self.melody_data.notes, chord_sequence):
            if pitch[0] in self.chord_mappings.get(chord, []):
                score += duration
        return score / self.melody_data.duration

    def _chord_variety(self, chord_sequence):
        unique_chords = len(set(chord_sequence))
        total_chords = len(self.chord_mappings)
        return unique_chords / total_chords

    def _harmonic_flow(self, chord_sequence):
        score = 0
        for i in range(len(chord_sequence) - 1):
            next_chord = chord_sequence[i + 1]
            if next_chord in self.preferred_transitions[chord_sequence[i]]:
                score += 1
        return score / (len(chord_sequence) - 1)

    def _functional_harmony(self, chord_sequence):
        score = 0
        if chord_sequence[0] in ["C", "Am"]:
            score += 1
        if chord_sequence[-1] in ["C"]:
            score += 1
        if "F" in chord_sequence and "G" in chord_sequence:
            score += 1
        return score / 3


def create_score(melody, chord_sequence, chord_mappings):
    score = music21.stream.Score()
    melody_part = music21.stream.Part()
    for note_name, duration in melody:
        melody_note = music21.note.Note(note_name, quarterLength=duration)
        melody_part.append(melody_note)

    chord_part = music21.stream.Part()
    current_duration = 0
    for (note_name, duration), chord_name in zip(melody, chord_sequence):
        chord_notes_list = chord_mappings.get(chord_name, [])
        chord_notes = music21.chord.Chord(
            chord_notes_list, quarterLength=duration
        )
        chord_notes.offset = current_duration
        chord_part.append(chord_notes)
        current_duration += duration

    score.append(melody_part)
    score.append(chord_part)
    return score


def main():
    twinkle_twinkle_melody = [
        ("C5", 1), ("C5", 1), ("G5", 1), ("G5", 1), ("A5", 1), ("A5", 1), ("G5", 2),
        ("F5", 1), ("F5", 1), ("E5", 1), ("E5", 1), ("D5", 1), ("D5", 1), ("C5", 2),
        ("G5", 1), ("G5", 1), ("F5", 1), ("F5", 1), ("E5", 1), ("E5", 1), ("D5", 2),
        ("G5", 1), ("G5", 1), ("F5", 1), ("F5", 1), ("E5", 1), ("E5", 1), ("D5", 2),
        ("C5", 1), ("C5", 1), ("G5", 1), ("G5", 1), ("A5", 1), ("A5", 1), ("G5", 2),
        ("F5", 1), ("F5", 1), ("E5", 1), ("E5", 1), ("D5", 1), ("D5", 1), ("C5", 2),
    ]
    weights = {
        "chord_melody_congruence": 0.5,
        "chord_variety": 0.3,
        "harmonic_flow": 0.1,
        "functional_harmony": 0.1
    }
    chord_mappings = {
        "C": ["C", "E", "G"],
        "Dm": ["D", "F", "A"],
        "Em": ["E", "G", "B"],
        "F": ["F", "A", "C"],
        "G": ["G", "B", "D"],
        "Am": ["A", "C", "E"],
        "Bdim": ["B", "D", "F"]
    }
    preferred_transitions = {
        "C": ["G", "Am", "F"],
        "Dm": ["G", "Am"],
        "Em": ["Am", "F", "C"],
        "F": ["C", "G"],
        "G": ["Am", "C"],
        "Am": ["Dm", "Em", "F", "C"],
        "Bdim": ["F", "Am"]
    }

    melody_data = MelodyData(twinkle_twinkle_melody)
    fitness_evaluator = FitnessEvaluator(
        melody_data=melody_data,
        weights=weights,
        chord_mappings=chord_mappings,
        preferred_transitions=preferred_transitions,
    )
    harmonizer = GeneticMelodyHarmonizer(
        melody_data=melody_data,
        chords=list(chord_mappings.keys()),
        population_size=100,
        mutation_rate=0.05,
        fitness_evaluator=fitness_evaluator,
    )

    generated_chords = harmonizer.generate(generations=1000)
    music21_score = create_score(
        twinkle_twinkle_melody, generated_chords, chord_mappings
    )
    music21_score.show()


if __name__ == "__main__":
    main()
