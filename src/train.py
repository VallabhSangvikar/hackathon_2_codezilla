import torch
from torch.utils.data import DataLoader
from preprocess import load_segmented_data
from utils import get_all_sessions, create_batches
from model import SpeakerSeparationModel

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpeakerSeparationModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    sessions = get_all_sessions('../data/')
    
    for epoch in range(50):
        model.train()
        total_loss = 0
        batch_count = 0
        
        for session in sessions:
            try:
                segments, clean, sr = load_segmented_data(session)
                batches = create_batches(segments, clean, batch_size=32)
                
                for mixed_batch, clean_batch in batches:
                    if mixed_batch.size(0) == 0:
                        continue
                    
                    # Print shapes for debugging
                    print(f"Mixed batch shape: {mixed_batch.shape}")
                    print(f"Clean batch shape: {clean_batch.shape}")
                    
                    mixed_batch = mixed_batch.to(device)
                    clean_batch = clean_batch.to(device)
                    
                    optimizer.zero_grad()
                    output = model(mixed_batch)
                    
                    # Ensure output and target have same dimensions
                    if output.size() != clean_batch.size():
                        print(f"Shape mismatch - Output: {output.size()}, Target: {clean_batch.size()}")
                        continue
                    
                    loss = model.loss_function(output, clean_batch)
                    
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    batch_count += 1
                    
                    if batch_count % 10 == 0:
                        print(f"Epoch {epoch}, Batch {batch_count}, Loss: {loss.item():.4f}")
                        
            except Exception as e:
                print(f"Error processing session {session}: {str(e)}")
                print(f"Shapes - Mixed: {mixed_batch.shape}, Clean: {clean_batch.shape}, Output: {output.shape}")
                continue
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    train_model()