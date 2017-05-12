import librosa
import numpy as np
from aupyom import Sampler, Sound

class AudioUtils:
    @classmethod
    def sub_sample_audio(cls, audio_file, out_audio_file, sample_rate=16000):
        y1, sr1 = librosa.load(audio_file, sr=sample_rate)
        librosa.output.write_wav(out_audio_file, y1, sample_rate, norm=True)

    @classmethod
    def segmentation_audio(cls, audio_file, out_audio_file, sample_rate=16000, begin_sec=0, split_duration=1):
        y1, sr1 = librosa.load(audio_file, sr=sample_rate)
        begin = int(sample_rate * begin_sec)
        y1 = y1[begin:begin + int(sample_rate * split_duration)]

        librosa.output.write_wav(out_audio_file, y1, sample_rate, norm=True)

    @classmethod
    def pitch_shift_audio_file(cls, audio_file, out_audio_file_path):
        y, sr = librosa.load(audio_file)
        y = librosa.effects.pitch_shift(y, sr, 10)
        librosa.output.write_wav(out_audio_file_path, y, sr)





    # @classmethod
    # def alignment_audio(cls, left_audio_file, right_audio_file, out_left_audio_file, out_right_audio_file,
    #                     sample_rate=16000):
    #     window_size = int(sample_rate * 0.1)
    #     y1, sr1 = librosa.load(left_audio_file, sr=sample_rate)
    #     y2, sr2 = librosa.load(right_audio_file, sr=sample_rate)
    #     output_y1 = []
    #     output_y2 = []
    #     len_size = min(len(y1), len(y2))
    #
    #     for start in range(0, len_size, window_size):
    #         if start + window_size >= len_size:
    #             break
    #         array1 = y1[start: start + window_size]
    #         array2 = y2[start: start + window_size]
    #         array1, array2 = MFCCAnalysis.align_data_two_arrays(array1, array2, int(0.004 * sample_rate), 0, 200)
    #         # print(array1, array2)
    #         output_y1.extend(array1)
    #         output_y2.extend(array2)
    #     output_y1 = np.asarray(output_y1)
    #     output_y2 = np.asarray(output_y2)
    #     librosa.output.write_wav(out_left_audio_file, output_y1, sample_rate, norm=True)
    #     librosa.output.write_wav(out_right_audio_file, output_y2, sample_rate, norm=True)
