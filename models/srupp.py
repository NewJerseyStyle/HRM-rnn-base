import torch
from torch import nn
from models.local_sru import SRUpp

class SRUpp_(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.sru = SRUpp(
            input_size=hidden_size,
            hidden_size=hidden_size,
            proj_size=hidden_size // 4,  # Use 1/4 of hidden size for projection
            num_layers=num_layers,
            dropout=0.1,  # Changed from rnn_dropout to dropout
            attn_dropout=0.1,
            num_heads=4,  # Use 4 attention heads
            layer_norm=True,
            normalize_after=False
        )

    def forward(self, x, hidden_state=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            hidden_state: Optional hidden state
        Returns:
            output: Output tensor
            new_hidden: New hidden state
        """
        # SRUpp expects (seq_len, batch_size, hidden_size) format
        x = x.transpose(0, 1)
        
        # Run SRU - SRUpp returns (output, hidden_state, extras)
        output, new_hidden, extras = self.sru(x, hidden_state)
        
        # Transpose back to (batch_size, seq_len, hidden_size)
        output = output.transpose(0, 1)
        
        return output, new_hidden
