from torch import nn


class Adapter(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size=None,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = self.input_size // 2 if hidden_size is None else hidden_size

        self.proj = nn.Sequential(
            nn.LayerNorm(self.input_size),
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.input_size),
        )

    def forward(self, x):
        return self.proj(x) + x
