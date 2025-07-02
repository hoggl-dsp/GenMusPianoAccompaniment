import unittest

from geneticmelodyharmonizer_GBT2 import chord_strings_to_midi_chords, note_string_to_midi_note


class TestChordStringsToMidiChords(unittest.TestCase):

    def test_01(self):
        c_mayor_chord = ['C', 'E', 'G']
        c_minor_chord = ['C', 'Eb', 'G']
        c_dim_chord = ['C', 'Eb', 'Gb']
        c_aug_chord = ['C', 'E', 'G#']

        c_mayor_chord = [
            note_string_to_midi_note(note) for note in c_mayor_chord
        ]
        c_minor_chord = [
            note_string_to_midi_note(note) for note in c_minor_chord
        ]
        c_dim_chord = [note_string_to_midi_note(note) for note in c_dim_chord]
        c_aug_chord = [note_string_to_midi_note(note) for note in c_aug_chord]

        result = chord_strings_to_midi_chords(['C', 'Cm', 'Cdim', 'Caug'])

        self.assertListEqual(
            result, [c_mayor_chord, c_minor_chord, c_dim_chord, c_aug_chord])
        
        # for chord, expected_chord in zip(['C', 'Cm', 'Cdim', 'Caug'], [c_mayor_chord, c_minor_chord, c_dim_chord, c_aug_chord]):
        #     self.assertEqual(chord_strings_to_midi_chords([chord]), [expected_chord])

class TestNote_string_to_midi_note(unittest.TestCase):
    def test_01(self):
        self.assertEqual(note_string_to_midi_note('G#'), 68)

if __name__ == '__main__':
    unittest.main()
