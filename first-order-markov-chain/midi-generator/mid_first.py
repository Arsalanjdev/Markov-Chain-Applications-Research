import os
import random
from mido import MidiFile, MidiTrack, Message


def build_transition_matrix(mid_files_folder):
    transition_matrix = {}

    for filename in os.listdir(mid_files_folder):
        if filename.endswith(".mid"):
            midi_file = MidiFile(os.path.join(mid_files_folder, filename))
            notes = []

            for track in midi_file.tracks:
                for msg in track:
                    if msg.type == 'note_on':
                        notes.append(msg.note)

            for i in range(len(notes) - 1):
                current_note = notes[i]
                next_note = notes[i + 1]

                if current_note not in transition_matrix:
                    transition_matrix[current_note] = {}

                if next_note not in transition_matrix[current_note]:
                    transition_matrix[current_note][next_note] = 1
                else:
                    transition_matrix[current_note][next_note] += 1

    for current_note in transition_matrix:
        total_occurrences = sum(transition_matrix[current_note].values())
        for next_note in transition_matrix[current_note]:
            transition_matrix[current_note][next_note] /= total_occurrences

    return transition_matrix


def generate_midi(transition_matrix, output_file, num_notes=100):
    midi_file = MidiFile()
    track = MidiTrack()
    midi_file.tracks.append(track)

    current_note = random.choice(list(transition_matrix.keys()))

    for _ in range(num_notes):
        track.append(Message('note_on', note=current_note, velocity=64, time=0))
        track.append(
            Message('note_off', note=current_note, velocity=64, time=500))  # Adjust the time for a slower tempo

        if current_note in transition_matrix:
            next_notes = list(transition_matrix[current_note].keys())
            probabilities = list(transition_matrix[current_note].values())
            current_note = random.choices(next_notes, probabilities)[0]
        else:
            break

    midi_file.save(output_file)


mid_files_folder = "midi"
output_file = "generated.mid"
transition_matrix = build_transition_matrix(mid_files_folder)
generate_midi(transition_matrix, output_file, num_notes=100)
