from train_model import train_model
from infer_audio import infer_audio

if __name__ == "__main__":
    # Step 1: Train the model
    data_dir = "./dataset"
    model_save_path = "./model/speaker_diarization_model.h5"
    train_model(data_dir, model_save_path)

    # Step 2: Test the model
    mixed_audio_path = "./mixed_audio/test_mixed_audio.wav"
    output_dir = "./output"
    infer_audio(mixed_audio_path, model_save_path, output_dir)
