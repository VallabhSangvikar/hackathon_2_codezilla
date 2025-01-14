# src/preprocess.py
import librosa
import numpy as np
import torch
import os

# Define constants
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

def load_segmented_data(session_path, segment_length=2496):  # Fixed length for all segments
    segments_path = os.path.join(session_path, 'record', 'segments')
    clean_path = os.path.join(session_path, 'clean')
    
    segment_files = sorted([f for f in os.listdir(segments_path) if f.endswith('.wav')])
    clean_files = sorted([f for f in os.listdir(clean_path) if f.endswith('.wav')])
    
    segments_data = []
    clean_data = []
    
    for seg_file in segment_files:
        audio, sr = librosa.load(os.path.join(segments_path, seg_file))
        # Ensure fixed length through padding or truncation
        if len(audio) < segment_length:
            audio = np.pad(audio, (0, segment_length - len(audio)))
        else:
            audio = audio[:segment_length]
            
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=sr, 
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )
        mel_spec = librosa.power_to_db(mel_spec)
        # Ensure consistent dimensions
        mel_spec = torch.FloatTensor(mel_spec).transpose(0, 1)  # [time, freq]
        segments_data.append(mel_spec)
    
    for clean_file in clean_files:
        audio, sr = librosa.load(os.path.join(clean_path, clean_file))
        if len(audio) < segment_length:
            audio = np.pad(audio, (0, segment_length - len(audio)))
        else:
            audio = audio[:segment_length]
            
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )
        mel_spec = librosa.power_to_db(mel_spec)
        # Ensure consistent dimensions
        mel_spec = torch.FloatTensor(mel_spec).transpose(0, 1)  # [time, freq]
        clean_data.append(mel_spec)
    
    return segments_data, clean_data, sr

# src/utils.py
def process_segment(audio_segment, sr):
    """
    Process each segment into features
    """
    # Convert to mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio_segment, sr=sr)
    return mel_spec

def get_segment_info(session_path):
    """
    Get transcription info for segments
    """
    trans_path = os.path.join(session_path, 'transcription', 'segments')
    segment_info = {}
    
    # Read segment transcription files
    for file in os.listdir(trans_path):
        if file.endswith('.txt'):
            with open(os.path.join(trans_path, file), 'r') as f:
                segment_info[file] = f.readlines()
    
    return segment_info

# src/train.py
def train_model():
    model = SpeakerSeparationModel()
    
    sessions = get_all_sessions('data/')
    
    for session in sessions:
        # Load segmented data
        segments, clean, sr = load_segmented_data(session)
        segment_info = get_segment_info(session)
        
        for segment in segments:
            features = process_segment(segment, sr)
            # Train model with these features
            # ... training code ...