import datetime
import re
import subprocess, sys
import os
import wave
import contextlib
from threading import Thread
import numpy as np

from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

from audio_utils import AudioUtils


class MFCCAnalysis:
    output_dir = 'output'
    file_index = 0
    split_duration = 2
    total_wav_filename = 'total.wav'
    threshold_coefficient = 3 / 5
    simple_move_average_length = 25

    @classmethod
    def get_dat_files(cls, ktv_dirname):
        path_files = []
        for dirname, dirnames, filenames in os.walk('.' + ktv_dirname):
            for filename in filenames:
                if re.search('.*\.dat', filename):
                    path_files.append(os.path.realpath(os.path.join(dirname, filename)))
        return path_files

    @classmethod
    def ffmpeg_extract_audio(cls, input_file_path, dir_name):
        relative_out_path = os.path.join(cls.output_dir, dir_name)
        if not os.path.exists(relative_out_path):
            os.makedirs(os.path.join(relative_out_path))
        cmd = "ffmpeg -i %s -vn %s/%s -y" % (input_file_path, relative_out_path, cls.total_wav_filename)
        return_code = subprocess.call(cmd, shell=True)

    @classmethod
    def get_duration(cls, file):
        with contextlib.closing(wave.open(file, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            return duration

    @classmethod
    def ffmpeg_extract_mono_stereo_and_split(cls, dir_name, side='left'):
        total_wav_file_path = os.path.join(cls.output_dir, dir_name)
        split_duration_format = str(datetime.timedelta(seconds=cls.split_duration))
        relative_out_path = os.path.join(cls.output_dir, dir_name)

        total_duration = cls.get_duration(os.path.join(total_wav_file_path, cls.total_wav_filename))
        current_timestamp = 0

        while current_timestamp < total_duration:
            current_timestamp_wav_filename = side + str(current_timestamp) + '.wav'
            channel = '0.0.0' if side == 'left' else '0.0.1'
            cmd = "ffmpeg -y -i %s/%s  -map_channel %s -ss %s -t %s %s/%s " % (
                total_wav_file_path,
                cls.total_wav_filename,
                channel,
                str(datetime.timedelta(seconds=current_timestamp)),
                split_duration_format,
                total_wav_file_path,
                current_timestamp_wav_filename,
            )
            current_timestamp += cls.split_duration
            return_code = subprocess.call(cmd, shell=True)

    @classmethod
    def get_mfcc_feature(cls, file_path):
        (rate, sig) = wav.read(file_path)
        return mfcc(sig, rate, appendEnergy=False, numcep=26)

    @classmethod
    def is_almost_same(cls, file_path1, file_path2):
        (rate, sig) = wav.read(file_path1)
        (rate2, sig2) = wav.read(file_path2)
        sub = np.abs(sig - sig2)
        for v in sub:
            if v > 10:
                return False
        return True

    @classmethod
    def auto_shift_and_minimize_error_wav(cls, base_file_path, adjust_file_path):
        (rate, sig) = wav.read(base_file_path)
        (rate2, sig2) = wav.read(adjust_file_path)
        shift = cls.count_array_shift_error(sig, sig2, int(rate * 0.02))
        return shift / rate

    @classmethod
    def minus_error(cls, base_file_path, adjust_file_path):
        (rate, sig) = wav.read(base_file_path)
        (rate2, sig2) = wav.read(adjust_file_path)
        array1 = np.asarray(sig)
        array2 = np.asarray(sig2)
        error_array = []
        for index in range(0, 30):
            error = np.sum(np.abs(array1[index * 4000:(index + 1) * 4000] - array2[index * 4000:(index + 1) * 4000]))
            error_array.append(error)
        plt.plot(error_array)
        plt.show()

    @classmethod
    def count_array_shift_error(cls, base_array, shifting_array, shift_count):
        shifting_array = np.asarray(shifting_array)
        base_array = np.asarray(base_array)
        min_error = sys.maxsize
        error_array = []
        shift = 0
        min_length = min(len(base_array), len(shifting_array))
        for shift_num in range(-shift_count, shift_count):
            # print(base_array, shifting_array)
            shifted_array = np.roll(shifting_array, shift_num)
            if shift_num < 0:
                error = np.sum(np.abs(base_array[:min_length] - shifted_array[:min_length])[:shift_num])
            else:
                error = np.sum(np.abs(base_array[:min_length] - shifted_array[:min_length])[shift_num:])
            if error < min_error:
                shift = shift_num
                min_error = error
            error_array.append(error)
        plt.plot(error_array)
        plt.show()
        return shift

    @staticmethod
    def correlation(array1, array2, shift_begin, shift_end, step, range_from, range_to, correction_value=10000000):
        sum_array = []
        for k in range(shift_begin, shift_end, step):

            arr1 = array1[range_from: range_to]
            arr2 = array2[range_from: range_to]
            if k >= 0:
                arr1 = np.roll(arr1, k)[k:]
                arr2 = arr2[k:]
            else:
                arr1 = np.roll(arr1, k)[:k]
                arr2 = arr2[:k]
            sum = np.sum(arr1 * np.around(np.multiply(arr2, correction_value)))

            # sum = np.sum(arr1 * np.around(np.multiply(arr2, 0.001)))
            # for index, val in enumerate(array1[range_from:range_to]):
            #     if index + k < 0 or index + k >= len(arr2):
            #         continue
            #     val2 = arr2[index + k]
            #     sum += val * int(val2 * 0.001)
            sum_array.append(sum)

        max_num = max(sum_array)
        max_index = [index for index, value in enumerate(sum_array) if value == max_num][0]
        return max_index, max_num, sum_array

    @staticmethod
    def correlation_audio(wav_file1, wav_file2):
        (rate, sig) = wav.read(wav_file1)
        (rate2, sig2) = wav.read(wav_file2)

        array1 = np.asarray(sig)
        array2 = np.asarray(sig2)
        max_index, max_num, sum_array = MFCCAnalysis.correlation(array1,
                                                                 array2,
                                                                 int(rate * -0.01),
                                                                 int(rate * 0.01),
                                                                 10,
                                                                 int(rate * 2.0),
                                                                 int(rate * 3.0))
        print(max_index)
        plt.plot(sum_array)
        plt.show()
        return max_index / rate

    @staticmethod
    def correlation_mfcc_coef_line(lines, lines2, coef_index=0):
        array1 = np.asarray(lines[coef_index])
        array2 = np.asarray(lines2[coef_index])
        max_index, max_num, sum_array = MFCCAnalysis.correlation(array1,
                                                                 array2,
                                                                 -100,
                                                                 100,
                                                                 1,
                                                                 100,
                                                                 200)
        print(max_index)
        plt.plot(sum_array)
        plt.show()

    @classmethod
    def compare_mfcc_from_stereo(cls, dir_name, show=False):
        dir_path = os.path.join(cls.output_dir, dir_name)
        current_compare_timestamp = 0
        right_file_path = os.path.join(dir_path, "right%s.wav" % current_compare_timestamp)
        left_file_path = os.path.join(dir_path, "left%s.wav" % current_compare_timestamp)
        plot_array = []
        while os.path.isfile(right_file_path) and os.path.isfile(left_file_path):
            current_compare_timestamp += cls.split_duration
            right_total_steps = cls.get_mfcc_feature(right_file_path)
            left_total_steps = cls.get_mfcc_feature(left_file_path)
            min_total_step = min(right_total_steps.shape[0], left_total_steps.shape[0])
            for i in range(0, min_total_step):
                if current_compare_timestamp == 3 and i > 3:
                    print(i)
                    print(left_total_steps[i])
                    print(right_total_steps[i])
                right_current_step = right_total_steps[i, 5:20]
                left_current_step = left_total_steps[i, 5:20]
                biggest_sum = np.sum(np.abs(left_current_step)) if np.sum(np.abs(left_current_step)) > np.sum(
                    np.abs(right_current_step))  else np.sum(np.abs(
                    right_current_step))
                result = (left_current_step - right_current_step) * (left_current_step - right_current_step)
                result = np.math.sqrt(np.sum(result))
                plot_array.append(result)

            right_file_path = os.path.join(dir_path, "right%s.wav" % current_compare_timestamp)
            left_file_path = os.path.join(dir_path, "left%s.wav" % current_compare_timestamp)

        plt.plot(plot_array)
        plt.ylim(ymax=100, ymin=0)
        plt.ylabel('Error')
        plt.xlabel('Time(ms)')
        plt.savefig(os.path.join(dir_path, dir_name + '.png'))
        if show:
            plt.show()
        plt.clf()

    @staticmethod
    def get_mfcc_coef_line(file_name):
        mfcc_matrix = MFCCAnalysis.get_mfcc_feature(file_name)
        mfcc_line_array = []
        for step in range(0, len(mfcc_matrix)):
            mfcc_array = mfcc_matrix[step]
            for index, val in enumerate(mfcc_array):
                if index not in mfcc_line_array:
                    mfcc_line_array.append([])
                mfcc_line_array[index].append(val)
        return mfcc_line_array

    @staticmethod
    def diff_mfcc_coef_line(file_name):
        lines = MFCCAnalysis.get_mfcc_coef_line(file_name)
        new_lines = []
        for line in lines:
            line = np.asarray(line)
            if len(line) == 0:
                break
            forward_line = line[1:]
            forward_line = np.append(forward_line, line[-1])
            new_lines.append(forward_line - line)

            # new_line = []
            # for index, value in enumerate(line):
            #     prev_index = index - 1
            #     if index <= 0:
            #         continue
            #     pre_value = line[prev_index]
            #     new_line.append(value - pre_value)
            # new_lines.append(new_line)
        return new_lines

    @staticmethod
    def error_with_diff_mfcc_coef_line(dir_name):
        current_timestamp = 0
        error_lines = []
        right_file_path = os.path.join(dir_name, "right%s.wav" % current_timestamp)
        left_file_path = os.path.join(dir_name, "left%s.wav" % current_timestamp)
        while os.path.isfile(right_file_path) and os.path.isfile(left_file_path):
            if current_timestamp >= 20:
                break
            print('current timestamp is %s' % current_timestamp)
            right_lines = MFCCAnalysis.diff_mfcc_coef_line(right_file_path)
            left_lines = MFCCAnalysis.diff_mfcc_coef_line(left_file_path)
            for right_line, left_line in zip(right_lines, left_lines):
                right_line = np.asarray(right_line)
                left_line = np.asarray(left_line)
                error_line = np.abs(right_line - left_line)
                error_lines.append(error_line)
            current_timestamp += 3
            right_file_path = os.path.join(dir_name, "right%s.wav" % current_timestamp)
            left_file_path = os.path.join(dir_name, "left%s.wav" % current_timestamp)
        return error_lines

    @staticmethod
    def get_samp_diff_from_audio_files(dir_name):
        current_timestamp = 0
        right_file_path = os.path.join(dir_name, "right%s.wav" % current_timestamp)
        left_file_path = os.path.join(dir_name, "left%s.wav" % current_timestamp)
        right_samp = 0
        left_samp = 0
        right_samp_count = 0
        left_samp_count = 0
        while os.path.isfile(right_file_path) and os.path.isfile(left_file_path):
            (rate, sig) = wav.read(right_file_path)
            (rate2, sig2) = wav.read(left_file_path)
            sig = np.asarray(sig)
            sig2 = np.asarray(sig2)

            right_samp += np.sum(np.abs(sig))
            left_samp += np.sum(np.abs(sig2))
            right_samp_count += len(sig)
            left_samp_count += len(sig2)
            current_timestamp += 3
            right_file_path = os.path.join(dir_name, "right%s.wav" % current_timestamp)
            left_file_path = os.path.join(dir_name, "left%s.wav" % current_timestamp)

        right_abs = right_samp / right_samp_count
        left_abs = left_samp / left_samp_count
        diff = right_abs / left_abs
        return diff

    @staticmethod
    def align_data_two_arrays(array1, array2, shift, range_begin, range_end):
        max_len = max(len(array1), len(array2))
        # needed_shift = MFCCAnalysis.count_array_shift_error(array1, array2, shift)
        max_index, max_num, sum_array = MFCCAnalysis.correlation(array1, array2, -shift, shift, 1, range_begin,
                                                                 range_end)
        # plt.plot(sum_array)
        # plt.show()
        needed_shift = shift - max_index
        # print("shift %s max index %s" % (shift, max_index))
        if needed_shift < 0:
            array1 = np.roll(array1, -needed_shift)
            array1 = array1[-needed_shift:]
            array2 = array2[-needed_shift:]
        else:
            array2 = np.roll(array2, needed_shift)
            array2 = array2[needed_shift:]
            array1 = array1[needed_shift:]
        return array1, array2

    @staticmethod
    def error_from_audio_files(dir_name, window_size=0.1, start_timestamp=0):
        error_list = []
        samp_diff = MFCCAnalysis.get_samp_diff_from_audio_files(dir_name)

        current_timestamp = start_timestamp
        right_file_path = os.path.join(dir_name, "right%s.wav" % current_timestamp)
        left_file_path = os.path.join(dir_name, "left%s.wav" % current_timestamp)
        while os.path.isfile(right_file_path) and os.path.isfile(left_file_path):
            (rate, sig) = wav.read(right_file_path)
            (rate2, sig2) = wav.read(left_file_path)
            frames_windows = int(rate * window_size)
            sig = np.asarray(sig)
            sig2 = np.asarray(sig2)
            # sig2 = sig2 * samp_diff

            minlen = min(len(sig), len(sig2))
            for start in range(0, minlen, frames_windows):
                if start + frames_windows >= minlen:
                    break
                array1 = sig[start:start + frames_windows]
                array2 = sig2[start:start + frames_windows]
                # if current_timestamp==start_timestamp :
                #     plt.clf()
                #     plt.plot(array1)
                #     plt.plot(array2)
                #     plt.savefig('aligned_before.png')
                #     plt.clf()
                array1, array2 = MFCCAnalysis.align_data_two_arrays(array1, array2, int(0.004 * rate), 0, 500)
                # if current_timestamp==start_timestamp:
                #     plt.plot(array1)
                #     plt.plot(array2)
                #     plt.savefig('aligned_after.png')

                sum_error = np.sum(np.abs(array1 - array2))
                error_list.append(sum_error)

            current_timestamp += 3
            right_file_path = os.path.join(dir_name, "right%s.wav" % current_timestamp)
            left_file_path = os.path.join(dir_name, "left%s.wav" % current_timestamp)
            print("current timestamp %s" % current_timestamp)

        return MFCCAnalysis.move_average(error_list)

    @staticmethod
    def move_average(error_list):
        lst = []
        for index, error in enumerate(error_list):
            error_sum = 0
            count = 0
            for index2 in range(index, index + MFCCAnalysis.simple_move_average_length):
                if index2 >= len(error_list):
                    break
                error_sum += error_list[index2]
            lst.append(error_sum)
        return lst

    @staticmethod
    def mean_threshold(error_list):
        pure_error_list = error_list[:]
        average = np.average(pure_error_list)
        indexes = []
        for index, error in enumerate(error_list):
            if error / average <= 0.07:
                print(index)
                indexes.append(index)
        pure_error_list = np.delete(pure_error_list, indexes)
        max_error = np.amax(pure_error_list)
        min_error = np.amin(pure_error_list)

        mean_error = (max_error + min_error) * MFCCAnalysis.threshold_coefficient
        return mean_error

    @staticmethod
    def over_threshold_wav(total_file, threshold, error_list, window_size=0.1, out_directory='./'):
        (rate, sig) = wav.read(total_file)
        new_sig = []
        new_sig2 = []
        for index, error in enumerate(error_list):
            if error >= threshold:
                begin_frame = int(index * window_size * rate)
                end_frame = int((index + 1) * window_size * rate)
                new_sig.extend(sig[begin_frame:end_frame])
            else:
                begin_frame = int(index * window_size * rate)
                end_frame = int((index + 1) * window_size * rate)
                new_sig2.extend(sig[begin_frame:end_frame])
        new_sig = np.asarray(new_sig)
        new_sig2 = np.asarray(new_sig2)
        wav.write(out_directory + '/result1.wav', rate, new_sig)
        wav.write(out_directory + '/result2.wav', rate, new_sig2)


        #
        # for index, value in enumerate(sig):
        #     value2 = sig2[min(len(sig2) - 1, index)]
        #     current_error += abs(value2 - value)
        #     if current_frames >= frames_windows:
        #         error_list.append(current_error)
        #         current_error = 0
        #         current_frames = 0
        #     current_frames += 1


    @staticmethod
    def subsample_folder(directory, sample_rate=16000):
        for root, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                file = os.path.join(root, filename)
                print(file)
                AudioUtils.sub_sample_audio(file, file, sample_rate)

    @staticmethod
    def clear_total_wav(directory):
        for root, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                file = os.path.join(root, filename)
                basename = os.path.basename(filename)
                name, ext = os.path.splitext(basename)
                if name == 'total':
                    os.remove(file)

    @staticmethod
    def run(file_path, dir_name):
        MFCCAnalysis.ffmpeg_extract_audio(file_path, dir_name)
        MFCCAnalysis.ffmpeg_extract_mono_stereo_and_split(dir_name, 'left')
        MFCCAnalysis.ffmpeg_extract_mono_stereo_and_split(dir_name, 'right')

        # MFCCAnalysis.compare_mfcc_from_stereo(dir_name)

# MFCCAnalysis.compare_mfcc_from_stereo('1', show=True)
