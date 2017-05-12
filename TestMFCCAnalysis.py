from unittest import TestCase
from mfcc_analysis import *
import numpy as np


class TestMFCCAnalysis(TestCase):
    def test_get_mfcc_feature(self):

        self.fail()

    def test_is_almost_same(self):
        self.failIf(not MFCCAnalysis.is_almost_same('test_data/same.wav', 'test_data/same2.wav'))
        self.failIf(MFCCAnalysis.is_almost_same('test_data/different.wav', 'test_data/different2.wav'))

    def test_auto_shift_and_minimize_error_wav(self):
        # shift = MFCCAnalysis.auto_shift_and_minimize_error_wav('test_data/right.wav', 'test_data/left.wav')
        # print(shift)
        # self.failUnlessAlmostEqual(0.0101587, shift)
        # self.fail()

        shift = MFCCAnalysis.auto_shift_and_minimize_error_wav('test_data/same2.wav', 'test_data/same.wav')
        print(shift)

    def test_count_array_shift_error(self):
        self.failUnlessEqual(MFCCAnalysis.count_array_shift_error([0, 1, 4, 9], [1, 4, 9, 4], 3), 1)
        self.failUnlessEqual(MFCCAnalysis.count_array_shift_error([0, -1, -4, -9], [-1, -4, -9, -4], 3), 1)
        self.failUnlessEqual(MFCCAnalysis.count_array_shift_error([0, 0, -1, -4, -9], [-1, -4, -9, -4, 0], 3), 2)
        self.failUnlessEqual(MFCCAnalysis.count_array_shift_error([0, 0, -1, -4, -9, 10], [-1, -4, -9, 10, 0], 3), 2)

    def test_correlation_audio(self):
        # sum_array = MFCCAnalysis.correlation_audio('test_data/left.wav', 'test_data/right.wav')
        # print(sum_array)
        sum_array = MFCCAnalysis.correlation_audio('test_data/same.wav', 'test_data/same2.wav')
        plt.plot(sum_array)
        plt.show()
        print(sum_array)

    def test_mfcc_left(self):
        lines = MFCCAnalysis.get_mfcc_coef_line('test_data/same.wav')
        for line in lines[:4]:
            plt.plot(line[:20])
        plt.show()

    def test_mfcc_right(self):
        lines = MFCCAnalysis.get_mfcc_coef_line('test_data/same2.wav')
        for line in lines[:4]:
            plt.plot(line)
        plt.show()

    def test_correlation(self):
        array1 = [0.1, 2, 3, 4, 5]
        array2 = [1, 2, 3, 4, 5, 0]
        max_index, max_num, coef_lines = MFCCAnalysis.correlation(array1, array2, -1, 1, 1, 1, 5)
        print(coef_lines)

    def test_correlation_mfcc_coef_line(self):
        lines = MFCCAnalysis.get_mfcc_coef_line('test_data/right3.wav')
        lines2 = MFCCAnalysis.get_mfcc_coef_line('test_data/left3.wav')
        # plt.plot(lines[0])
        # plt.plot(lines2[0])
        # plt.show()
        MFCCAnalysis.correlation_mfcc_coef_line(lines, lines2)

    def test_mfcc_diff_left(self):
        lines = MFCCAnalysis.diff_mfcc_coef_line('test_data/same.wav')
        print(len(lines))
        for line in lines[2:5]:
            plt.plot(line[:20])
        plt.show()

    def test_mfcc_diff_right(self):
        lines = MFCCAnalysis.diff_mfcc_coef_line('test_data/same2.wav')
        for line in lines[2:5]:
            plt.plot(line[:20])
        plt.show()

    def test_mfcc_diff_error(self):
        lines = MFCCAnalysis.error_with_diff_mfcc_coef_line('test_data/music')
        for line in lines[:3]:
            plt.plot(line)
        plt.show()

    def test_error_sum(self):
        for i in range(8, 9):
            print("processing %sth audio..." % i)
            directory = 'test_data/output/%s' % i
            # MFCCAnalysis.over_threshold_wav(directory + '/total.wav', 0, [], directory)
            error_list = MFCCAnalysis.error_from_audio_files(directory)
            mean_error = MFCCAnalysis.mean_threshold(error_list)
            MFCCAnalysis.over_threshold_wav(directory + '/total.wav', mean_error, error_list, out_directory=directory)
            plt.plot(error_list)
            plt.plot([0, len(error_list)], [mean_error, mean_error])
            plt.savefig(directory + '/fig.png')
            plt.clf()

    def test_error_sum_from_one_file(self):
        MFCCAnalysis.error_from_audio_files('output/', start_timestamp=132)



    def test_align_data_two_arrays(self):
        array1 = [1, 2, 0, 0]
        array2 = [0, 0, 1, 2]
        array1, array2 = MFCCAnalysis.align_data_two_arrays(array1, array2, 5, 0, len(array1))
        self.failIf(not np.array_equal(array1, [1, 2]))
        self.failIf(not np.array_equal(array2, [1, 2]))

        array1 = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 0, 0, 0]
        array2 = [0, 0, 0, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
        array1, array2 = MFCCAnalysis.align_data_two_arrays(array1, array2, 5, 0, len(array1))
        self.failIf(not np.array_equal(array1, [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]))
        self.failIf(not np.array_equal(array2, [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]))
        # print(array1, array2)


    def test_test(self):
        MFCCAnalysis.minus_error('test_data/same.wav', 'test_data/same2.wav')
