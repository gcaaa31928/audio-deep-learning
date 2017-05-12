

import numpy as np
# import scipy.io.wavfile
from mfcc_analysis import *
# from scikits.talkbox.features import mfcc
#
# sample_rate, X = scipy.io.wavfile.read("right0.wav")
# ceps, mspec, spec = mfcc(X)
# sample_rate2, X2 = scipy.io.wavfile.read("left0.wav")
# ceps2, mspec2, spec2= mfcc(X2)
# print(ceps[300])
# print(ceps2[300])
#

right = MFCCAnalysis.get_mfcc_feature("right0.wav")
left = MFCCAnalysis.get_mfcc_feature("left0.wav")
print(right[0])
print(left[0])
