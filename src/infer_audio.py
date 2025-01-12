import librosa
import numpy as np
import soundfile as sf
from tensorflow.keras.models import load_model

def extract_features_from_audio(chunk, n_mfcc=13, max_pad_len=1000):
    """Extract MFCC features from an audio chunk, matching the training process."""
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=chunk, sr=16000, n_mfcc=n_mfcc)
    
    # Pad or truncate to fixed length (matching training)
    pad_width = max_pad_len - mfcc.shape[1]
    if pad_width > 0:
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    
    return mfcc

def chunk_audio(audio_file_path, chunk_duration=0.1):
    """Chunk the audio file into segments."""
    y, sr = librosa.load(audio_file_path, sr=16000)
    chunks = []
    chunk_length = int(chunk_duration * sr)
    num_chunks = len(y) // chunk_length
    
    for i in range(num_chunks):
        start = i * chunk_length
        end = (i + 1) * chunk_length
        chunk = y[start:end]
        chunks.append(chunk)
    
    return chunks

def classify_speaker_for_chunks(model, audio_file_path):
    """Classify speaker for each audio chunk."""
    chunks = chunk_audio(audio_file_path)
    
    for i, chunk in enumerate(chunks):
        # Extract features matching the training process
        features = extract_features_from_audio(chunk)
        # Add batch and channel dimensions to match model's expected input shape
        features = np.expand_dims(features, axis=0)  # Add batch dimension
        features = np.expand_dims(features, axis=-1)  # Add channel dimension
        
        # Predict speaker
        prediction = model.predict(features, verbose=0)
        speaker = np.argmax(prediction)
        
        print(f"Speaker for chunk {i + 1}: {speaker}")
        # sf.write(f"output_speaker_{speaker}_chunk_{i + 1}.wav", chunk, 16000)

# Main execution
if __name__ == "__main__":
    # Load the model
    model = load_model('../model/speaker_diarization_model.h5')
    
    # Path to your test audio file
    audio_file_path = '../dataset/A/atharv4.wav'
    classify_speaker_for_chunks(model, audio_file_path)