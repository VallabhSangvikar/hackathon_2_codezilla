# src/utils.py
import os
import torch
from torch.nn.utils.rnn import pad_sequence

def get_all_sessions(base_path):
    """
    Get paths for all sessions in OV10 and OV20
    """
    sessions = []
    for overlap in ['OV10', 'OV20','OV30','OV40','0L','0S']:
        overlap_path = os.path.join(base_path, overlap)
        for session in os.listdir(overlap_path):
            if session.startswith('overlap_ratio'):
                sessions.append(os.path.join(overlap_path, session))
    return sessions

def create_batches(segments, clean, batch_size=32):
    """
    Create batches from segments and clean data
    """
    # Convert lists to tensors if they aren't already
    if not isinstance(segments[0], torch.Tensor):
        segments = [torch.FloatTensor(s) for s in segments]
    if not isinstance(clean[0], torch.Tensor):
        clean = [torch.FloatTensor(c) for c in clean]
    
    # Stack instead of pad since we now have fixed size
    segments = torch.stack(segments)
    clean = torch.stack(clean)
    
    # Create batches
    num_samples = segments.size(0)
    num_batches = max(1, num_samples // batch_size)
    
    batches = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        batch_segments = segments[start_idx:end_idx]
        batch_clean = clean[start_idx:end_idx]
        
        # Add channel dimension if not present
        if batch_segments.dim() == 3:
            batch_segments = batch_segments.unsqueeze(1)
        if batch_clean.dim() == 3:
            batch_clean = batch_clean.unsqueeze(1)
        
        batches.append((batch_segments, batch_clean))
    
    return batches

# Example usage:
base_path = "../data/"
all_sessions = get_all_sessions(base_path)