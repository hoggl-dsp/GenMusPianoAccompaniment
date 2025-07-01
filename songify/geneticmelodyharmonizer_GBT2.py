import random
from dataclasses import dataclass
from preferred_chord_transitions import preferred_transitions, chord_mappings
import music21


@dataclass(frozen=True)
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
    duration: int = None  # Computed attribute
    number_of_notes: int = None  # Computed attribute

    def __post_init__(self):
        object.__setattr__(
            self, "duration", sum(duration for _, duration in self.notes)
        )
        object.__setattr__(self, "number_of_notes", len(self.notes))


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
            # # Print average score
            # average_fitness = sum(
            #     self.fitness_evaluator.evaluate(seq) for seq in self._population
            # ) / self.population_size
            # print(f"Average fitness in generation: {average_fitness:.2f}")
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
            self._generate_random_chord_sequence()
            for _ in range(self.population_size)
        ]

    def _generate_random_chord_sequence(self):
        """
        Generate a random chord sequence with as many chords as the numbers
        of bars in the melody.

        Returns:
            list: List of randomly generated chords.
        """
        return [
            random.choice(self.chords)
            for _ in range(self.melody_data.number_of_notes)
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

    def __init__(
        self, melody_data, chord_mappings, weights, preferred_transitions
    ):
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
        for i, (chord, (pitch, duration)) in enumerate(zip(chord_sequence, self.melody_data.notes)):
            start = max(0, i - 2)
            end = min(len(self.melody_data.notes), i + 3)
            window_notes = self.melody_data.notes[start:end]
            window_pitches = [note[0] for note in window_notes]
            
            if any(p in self.chord_mappings[chord] for p in window_pitches):
                score += duration
        
        return score / self.melody_data.duration

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


def main():

    data = [
        (57, 0.15000000596046448, 0.44999999999999996, 1.0), 
        (55, 0.6000000238418579, 0.15000000000000002, 1.0), 
        (57, 0.75, 0.05, 1.0), 
        (58, 0.800000011920929, 0.15000000000000002, 1.0), 
        (52, 0.949999988079071, 0.05, 1.0), 
        (50, 1.0, 0.3, 0.9999992847442627), 
        (52, 1.2999999523162842, 0.05, 1.0), 
        (54, 1.350000023841858, 0.15000000000000002, 1.0), 
        (51, 2.049999952316284, 0.2, 1.0), 
        (49, 2.25, 0.05, 1.0), 
        (47, 2.299999952316284, 0.1, 0.9999985694885254), 
        (52, 2.4000000953674316, 0.1, 1.0), 
        (51, 2.5, 0.05, 1.0), 
        (48, 2.549999952316284, 0.1, 1.0), 
        (47, 2.6500000953674316, 0.1, 1.0), 
        (46, 2.75, 0.05, 1.0), 
        (48, 2.9000000953674316, 0.15000000000000002, 1.0), 
        (47, 3.049999952316284, 0.05, 1.0),  
        (55, 3.200000047683716, 0.1, 1.0), 
        (49, 3.299999952316284, 0.05, 1.0), 
        (47, 3.450000047683716, 0.3, 1.0), 
        (48, 3.75, 0.15000000000000002, 1.0), 
        (53, 3.9000000953674316, 0.1, 1.0), 
        (44, 4.0, 0.15000000000000002, 0.4803946912288666), 
        (45, 4.150000095367432, 0.1, 1.0), 
        (46, 4.25, 0.05, 1.0), 
        (50, 4.300000190734863, 0.05, 1.0), 
        (52, 4.349999904632568, 0.1, 1.0), 
        (41, 4.550000190734863, 0.2, 0.9998537302017212), 
        (39, 4.75, 0.05, 0.9999998807907104), 
        (51, 5.599999904632568, 0.1, 1.0), 
        (53, 6.0, 0.1, 1.0), 
        (52, 6.099999904632568, 0.05, 1.0), 
        (51, 6.150000095367432, 0.05, 1.0), 
        (50, 6.199999809265137, 0.2, 0.9975391626358032), 
        (49, 6.400000095367432, 0.05, 0.9998569488525391), 
        (48, 6.449999809265137, 0.1, 1.0), 
        (47, 6.550000190734863, 0.1, 0.9991744160652161), 
        (43, 7.0, 0.15000000000000002, 1.0), 
        (42, 7.150000095367432, 0.1, 0.9995341300964355), 
        (41, 7.25, 0.05, 1.0), 
        (40, 7.300000190734863, 0.1, 1.0), 
        (42, 7.400000095367432, 0.1, 1.0), 
        (50, 7.599999904632568, 0.1, 0.9998584985733032)]
    
    input_melody = [ (note[0], note[2]) for note in data ]  # Extract pitches from the melody
    starts = [ note[1] for note in data ]  # Extract start times from the melody
 
    weights = {
        "chord_melody_congruence": 0.3,
        "chord_variety": 0.2,
        "harmonic_flow": 0.5,
    }

    # Instantiate objects for generating harmonization
    melody_data = MelodyData(input_melody)
    fitness_evaluator = FitnessEvaluator(
        melody_data=melody_data,
        weights=weights,
        chord_mappings=chord_mappings,
        preferred_transitions=preferred_transitions,
    )
    harmonizer = GeneticMelodyHarmonizer(
        melody_data=melody_data,
        chords=list(chord_mappings.keys()),
        population_size=100, #TODO: Change this to increase
        mutation_rate=0.05,
        fitness_evaluator=fitness_evaluator,
    )

    # Generate chords with genetic algorithm
    generated_chords = harmonizer.generate(generations=1000)

    # Render to music21 score and show it
    music21_score = create_score(
        input_melody, starts, generated_chords, chord_mappings
    )
    music21_score.show()


if __name__ == "__main__":
    main()
