import torch
from torch.nn.functional import normalize

class Model(torch.nn.Module):

    def __init__(self, num_inputs, num_outputs, dropout=.2) -> None:
        super().__init__()

        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, 100),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(100, num_outputs),
        )

    def forward(self, x):
        x_norm = normalize(x.float(), p=1.0, dim=0)
        return self.sequential(x_norm)