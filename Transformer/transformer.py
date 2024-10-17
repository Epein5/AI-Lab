from models import *

class Trasformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: ImputEmbedding, tgt_embed: ImputEmbedding, src_pos_emb:PositionalEncoding, tgt_pos_emb:PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos_emb = src_pos_emb
        self.tgt_pos_emb = tgt_pos_emb
        self.projection_layer = projection_layer
    
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos_emb(src)
        return self.encoder(src, src_mask)
    
    def decode(self,encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos_emb(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)


def build_transformer(src_vocab_size:int, trt_vocab_size:int, src_seq_len:int, tgt_seq_len:int, d_model:int= 512, N:int = 6, h:int = 8, dropout: float= 0.1, d_ff:int = 2048) -> Trasformer:

    src_embedding = ImputEmbedding(d_model, src_vocab_size)
    tgt_embedding = ImputEmbedding(d_model, trt_vocab_size)

    src_poistional_encoding = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_positional_encoding = PositionalEncoding(d_model, tgt_seq_len, dropout)

    encoder_blocks = []
    for _ in range(N):
        self_attentaion_block = MultiHeadAttentation(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_model*4, dropout)
        encoder_block = EncoderBlock(self_attentaion_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    
    decoder_blocks = []
    for _ in range(N):
        self_attentaion_block = MultiHeadAttentation(d_model, h, dropout)
        cross_attentaion_block = MultiHeadAttentation(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(self_attentaion_block, cross_attentaion_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    projection_layer = ProjectionLayer(d_model, trt_vocab_size)

    transformer =  Trasformer(encoder, decoder, src_embedding, tgt_embedding, src_poistional_encoding, tgt_positional_encoding, projection_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer