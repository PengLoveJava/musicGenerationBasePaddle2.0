import glob
import os
import pickle
import numpy as np
from music21 import converter, instrument, note, chord, stream

def createNotes(file_path):
    """
    从音乐文件中读取notes(音符)，例如A0、C3等
    :return: notes(音符)
    """
    notes = []
    for midi_file in glob.glob(os.path.join(file_path, '*mid')):
        # 读取音乐文件
        midi = converter.parse(midi_file)
        # 检测音乐中的乐器部分
        parts = instrument.partitionByInstrument(midi)
        if parts:
            # 有乐器，取乐器
            notes_to_parse = parts.parts[0].recurse()
        else:
            # 无乐器，纯音符
            notes_to_parse = midi.flat.notes
        # 区分音符、和弦
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                # 音符
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                # 和弦
                notes.append('.'.join(str(n) for n in element.normalOrder))
    # 第一次使用的时候要取消注释生成notes词典
    with open('notes', 'wb') as file:
        pickle.dump(notes, file)
    return notes

def createSeq(notes):
    """
    将音乐文件中读取的notes转成网络训练所需的Sequences(序列)
    :return: Sequences(序列)
    """
    # 音符数量
    num_pitch = len(set(notes))

    # 序列长度
    sequence_length = 100
    # 音符去重，准备构造字典
    pitch_names = sorted(set(item for item in notes))
    # 字典
    pitch_to_int = dict((pitch, num) for num, pitch in enumerate(pitch_names))

    inputs = []
    # 构建序列
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        inputs.append([pitch_to_int[char] for char in sequence_in])
    inputs = np.reshape(inputs, (len(inputs), sequence_length, 1))

    # 归一化
    inputs = (inputs - float(num_pitch) / 2) / (float(num_pitch) / 2)

    return inputs

def createMusic(generator):
    """
    将网络Generator生成的音乐转成.mid文件，使其能播放
    :return: None
    """
    offset = 0
    output_notes = []
    for data in generator:
        if ('.' in data) or data.isdigit():
            note_in_chord = data.split('.')
            notes = []
            for current_note in note_in_chord:
                new_note = note.Note(int(current_note))
                # 选择钢琴作为乐源，当然可以尝试其它的，例如Mandolin(曼陀林)等
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(data)
            new_note.offset = offset
            # 选择钢琴作为乐源，当然可以尝试其它的，例如Mandolin(曼陀林)等
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='output.mid')

def musicGenerator(predict):
    with open('notes', 'rb') as fp:
        notes = pickle.load(fp)

    pitch_names = sorted(set(item for item in notes))
    int_to_note = dict((number, note) for number, note in enumerate(pitch_names))
    pred_notes = [x * 14 + 14 for x in predict[0]]
    pred_notes = [int_to_note[int(x)] for x in pred_notes]

    createMusic(pred_notes)