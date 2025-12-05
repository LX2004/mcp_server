import torch
import torch.nn as nn
from torch.nn import functional as F
from net.transformer import PositionalEncoding, TransformerEncoder


class Predict_encoder(nn.Module):
    def __init__(
        self,
        nhead,
        layers,
        hidden_dim,
        latent_dim,
        embedding_dim,
        seq_len,
        probs,
        device="cuda",
    ):
        super(Predict_encoder, self).__init__()

        self.layers = layers               # number of Transformer layers
        self.embedding_dim = embedding_dim # embedding dimension
        self.seq_len = seq_len             # sequence length
        self.nhead = nhead                 # number of attention heads
        self.hidden_dim = hidden_dim       # feed-forward dimension
        self.probs = probs                 # dropout probability
        self.device = device
        self.latent_dim = latent_dim       # output latent dimension
        self.src_mask = None

        self.pos_encoder = PositionalEncoding(
            device=self.device, d_model=self.embedding_dim, max_len=self.seq_len
        )

        self.transformer_encoder = TransformerEncoder(
            num_layers=self.layers,
            input_dim=self.embedding_dim,
            num_heads=self.nhead,
            dim_feedforward=self.hidden_dim,
            dropout=self.probs,
        )

        # global attention to pool sequence dimension
        self.glob_attn_module = nn.Sequential(
            nn.Linear(self.embedding_dim, 1),
            nn.Softmax(dim=1),
        )

        self.fc1 = nn.Linear(self.embedding_dim, self.latent_dim)

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        Create causal mask for the Transformer encoder.

        Args:
            sz: sequence length

        Returns:
            Tensor of shape (S, S) with 0 on and -inf above the diagonal.
        """
        mask = torch.ones((sz, sz), device=self.device)
        mask = (torch.triu(mask) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def transformer_encoding(self, embedded_batch: torch.Tensor) -> torch.Tensor:
        """
        Run the Transformer encoder over embedded inputs.

        Args:
            embedded_batch: tensor of shape (B, S, E)

        Returns:
            Tensor of shape (B, S, E)
        """
        if self.src_mask is None or self.src_mask.size(0) != embedded_batch.size(1):
            self.src_mask = self._generate_square_subsequent_mask(
                embedded_batch.size(1)
            )

        pos_encoded_batch = self.pos_encoder(embedded_batch)
        output_embed = self.transformer_encoder(pos_encoded_batch, self.src_mask)
        return output_embed

    def encoder(self, embedded_batch: torch.Tensor) -> torch.Tensor:
        """
        Encode embedded sequences into a latent representation.

        Args:
            embedded_batch: tensor of shape (B, S, E)

        Returns:
            Latent tensor of shape (B, latent_dim)
        """
        output_embed = self.transformer_encoding(embedded_batch)

        glob_attn = self.glob_attn_module(output_embed)  # (B, S, 1)
        z_rep = torch.bmm(glob_attn.transpose(-1, 1), output_embed).squeeze()

        # Restore batch dimension when B == 1
        if len(embedded_batch) == 1:
            z_rep = z_rep.unsqueeze(0)

        z_rep = self.fc1(z_rep)
        return z_rep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


if __name__ == "__main__":
    CUDA_LAUNCH_BLOCKING = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("*********device =", device, "*********")

    Pre_model = Predict_encoder(
        nhead=4,
        layers=4,
        hidden_dim=4,
        latent_dim=16,
        embedding_dim=100,
        seq_len=100,
        probs=0.1,
        device=device,
    ).to(device)

    z = torch.randn(size=(64, 100, 100), device=device)
    print(Pre_model(z).shape)
