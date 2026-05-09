import torch.nn as nn

from .utils.layer_norm import LayerNorm
from .attention import MultiHeadedAttention
from .utils import PositionwiseFeedForward


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        
        # sublayer 1 (Multi-Head Attention sublayer):
        self.layernorm1 = LayerNorm(hidden)
        self.multiheadedattention = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        
        # sublayer 2 (Position-wise Feed Forward sublayer):
        self.layernorm2 = LayerNorm(hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, mask):
        # sublayer 1 (Multi-Head Attention sublayer):
        x_skip_connection1 = x
        x = self.layernorm1(x)
        x = self.multiheadedattention(x, x, x, mask)
        x = self.dropout1(x)
        x += x_skip_connection1
        
        # sublayer 2 (Position-wise Feed Forward sublayer):
        x_skip_connection2 = x
        x = self.layernorm2(x)
        x = self.feed_forward(x)
        x = self.dropout2(x)
        x += x_skip_connection2
        
        return self.dropout3(x)
