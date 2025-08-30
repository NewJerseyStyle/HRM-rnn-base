from torch import nn
from models.local_sru import SRUpp

class SRUpp_(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.sru = SRUpp(hidden_size, hidden_size, proj_size=0, rnn_dropout=0.2, use_tanh=False, use_relu=True)

    def forward(self, x):
        x, _ = self.sru(x)
        return x
