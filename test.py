
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav

(rate,sig) = wav.read("english.wav")
mfcc_feat = mfcc(sig,rate)
fbank_feat = logfbank(sig,rate)
print(mfcc_feat)
print(len(fbank_feat[2]))

