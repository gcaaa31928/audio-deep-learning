from unittest import TestCase

from audio_utils import AudioUtils


class TestAudioUtils(TestCase):
    def test_sub_sample_audio(self):
        AudioUtils.sub_sample_audio('LizNelson_Rainfall_MIX.wav', 'sub_LizNelson_Rainfall_MIX.wav')
        AudioUtils.sub_sample_audio('LizNelson_Rainfall_RAW_01_01.wav', 'sub_LizNelson_Rainfall_RAW_01_01.wav')

    def test_segmentation_audio(self):
        AudioUtils.segmentation_audio('sub_LizNelson_Rainfall_MIX.wav', 'input.wav', begin_sec=16.5, split_duration=2)
        AudioUtils.segmentation_audio('sub_LizNelson_Rainfall_RAW_01_01.wav', 'output.wav', begin_sec=16.5,
                                      split_duration=2)

    def test_alignment_audio(self):
        AudioUtils.alignment_audio('no_human1.wav', 'human1.wav', 'right.wav', 'left.wav')
