import librosa
import numpy as np

def extract_features(audio_path, n_mfcc=13, max_pad_len=1000):
    """Extract MFCC features from an audio file."""
    audio, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    # Pad or truncate to fixed length
    pad_width = max_pad_len - mfcc.shape[1]
    if pad_width > 0:
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    return mfcc
