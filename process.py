from threading import Thread

from audio_utils import AudioUtils
from mfcc_analysis import MFCCAnalysis

def main():
    # thread_pools = []
    # dir_name_index = 0
    # for path_file in MFCCAnalysis.get_dat_files('/KTV'):
    #
    #     thread = Thread(target=MFCCAnalysis.run, args=(path_file, str(dir_name_index)))
    #     thread.start()
    #     thread_pools.append(thread)
    #     dir_name_index += 1
    #
    # for thread in thread_pools:
    #     thread.join()
    # MFCCAnalysis.subsample_folder('./output')
    # MFCCAnalysis.clear_total_wav('./output')

    AudioUtils.pitch_shift_audio_file('./input.wav', './test.wav')
    print('extract audio success')





if __name__ == '__main__':
    main()
