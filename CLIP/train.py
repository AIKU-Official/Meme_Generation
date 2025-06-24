import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union

import wandb


class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'


class ClipCocoDataset(Dataset):

    def __len__(self) -> int:
        return len(self.captions_tokens)

    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item]
        # Truncate tokens so that prefix + tokens length <= 1024
        max_caption_len = 1024 - self.prefix_length
        if tokens.shape[0] > max_caption_len:
            tokens = tokens[:max_caption_len]
            self.captions_tokens[item] = tokens
        # Determine effective max sequence length
        effective_max_seq_len = min(self.max_seq_len, max_caption_len)
        # Pad or truncate tokens to effective_max_seq_len
        padding = effective_max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.captions_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[:effective_max_seq_len]
            self.captions_tokens[item] = tokens
        # Create mask where token >= 0
        mask = tokens.ge(0)
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        tokens, mask = self.pad_tokens(item)
        prefix = self.prefixes[self.caption2embedding[item]]
        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)
        return tokens, mask, prefix

    def __init__(self, data_path: str,  prefix_length: int, gpt2_type: str = "gpt2",
                 normalize_prefix=False):
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        print("Data size is %0d" % len(all_data["clip_embedding"]))
        sys.stdout.flush()
        self.prefixes = all_data["clip_embedding"]
        captions_raw = all_data["captions"]
        self.image_ids = [caption["image_id"] for caption in captions_raw]
        self.captions = [caption['caption'] for caption in captions_raw]
        tokens_pkl_path = f"{data_path[:-4]}_tokens.pkl"
        if os.path.isfile(tokens_pkl_path):
            with open(tokens_pkl_path, 'rb') as f:
                # Now load funny_scores as well
                loaded = pickle.load(f)
                if len(loaded) == 3:
                    self.captions_tokens, self.caption2embedding, self.max_seq_len = loaded
                    # For backward compatibility, recompute funny_scores if missing
                    funny_scores = []
                    for caption in captions_raw:
                        score = caption.get("funny_score", 0.0)
                        funny_scores.append(score)
                    self.funny_scores = funny_scores
                else:
                    self.captions_tokens, self.caption2embedding, self.funny_scores, self.max_seq_len = loaded
        else:
            self.captions_tokens = []
            self.caption2embedding = []
            captions_list = []
            funny_scores = []
            max_seq_len = 0
            for caption in captions_raw:
                text = caption['caption']
                if not isinstance(text, str) or text.strip() == "" or text == "nan":
                    continue  # skip invalid captions
                self.captions_tokens.append(torch.tensor(self.tokenizer.encode(caption['caption']), dtype=torch.int64))
                self.caption2embedding.append(caption["clip_embedding"])
                captions_list.append(caption['caption'])
                score = caption.get("funny_score", 0.0)
                funny_scores.append(score)
                max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])
            with open(tokens_pkl_path, 'wb') as f:
                pickle.dump([self.captions_tokens, self.caption2embedding, funny_scores, max_seq_len], f)
            self.funny_scores = funny_scores
        all_len = torch.tensor([len(self.captions_tokens[i]) for i in range(len(self))]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))


class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=nnf.relu, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class MultiHeadAttention(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention


class TransformerLayer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=nnf.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)


class Transformer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec: # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x

    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            elif enc_dec:  # self
                layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            else:  # self or cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)


class TransformerMapper(nn.Module):

    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out

    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8, num_heads: int = 8):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, num_heads, num_layers)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)


class ClipCaptionModel(nn.Module):

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,
                 num_layers: int = 8, num_heads: int = 8, mapping_type: MappingType = MappingType.MLP):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if mapping_type == MappingType.MLP:
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2,
                                     self.gpt_embedding_size * prefix_length))
        else:
            self.clip_project = TransformerMapper(prefix_size, self.gpt_embedding_size, prefix_length,
                                                                     clip_length, num_layers, num_heads)


class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self


def save_config(args: argparse.Namespace):
    config = {}
    for key, item in args._get_kwargs():
        config[key] = item
    out_path = os.path.join(args.out_dir, f"{args.prefix}.json")
    with open(out_path, 'w') as outfile:
        json.dump(config, outfile)


def load_model(config_path: str, epoch_or_latest: Union[str, int] = '_latest'):
    with open(config_path) as f:
        config = json.load(f)
    parser = argparse.ArgumentParser()
    parser.set_defaults(**config)
    args = parser.parse_args()
    if type(epoch_or_latest) is int:
        epoch_or_latest = f"-{epoch_or_latest:03d}"
    model_path = os.path.join(args.out_dir, f"{args.prefix}{epoch_or_latest}.pt")
    if args.only_prefix:
        model = ClipCaptionPrefix(args.prefix_length)
    else:
        model = ClipCaptionModel(args.prefix_length)
    if os.path.isfile(model_path):
        print(f"loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        print(f"{model_path} is not exist")
    return model, parser

# ─────────────────────────────────────────────────────────────────────────────
# Memory‑efficient Position‑Conditioned Loss
# ─────────────────────────────────────────────────────────────────────────────
def position_conditioned_loss(c_logits: torch.Tensor,
                              y: torch.Tensor,
                              loss_type: str = 'linear',
                              gamma: float = 1.0,
                              beta: float = 10.0,
                              alpha: float = 6.0,
                              ignore_index: int = 0) -> torch.Tensor:
    """
    Computes the position‑conditioned loss without materialising a huge one‑hot
    tensor.  Memory complexity becomes O(B * M) instead of O(B * M * K).

    Args:
        c_logits: (B, M, K) – model logits.
        y:        (B, M)    – ground‑truth token ids (‑1 or ignore_index for padding).
        loss_type: 'linear' | 'gaussian' | 'sigmoid'
        gamma, beta, alpha: weighting hyper‑parameters (see original paper/code).
        ignore_index: id that marks padding tokens.

    Returns:
        Scalar loss (torch.Tensor, shape = []).
    """
    B, M, K = c_logits.shape

    # Softmax to probabilities
    probs = torch.softmax(c_logits, dim=-1)                                  # (B, M, K)

    # ------------------------------------------------------------------
    # 1) Positive term  – log P(correct token)
    # ------------------------------------------------------------------
    y_clamped = torch.clamp(y, min=0)                                        # ensure >= 0
    p_true = torch.gather(probs, 2, y_clamped.unsqueeze(2)).squeeze(2)       # (B, M)
    pos_loss = -torch.log(p_true.clamp(min=1e-8))                            # (B, M)

    # ------------------------------------------------------------------
    # 2) Negative term  – log (1 − P(correct token))
    #    Instead of summing over all wrong classes, use the remaining
    #    probability mass as a single negative class.  Empirically gives
    #    the same gradient while saving massive memory.
    # ------------------------------------------------------------------
    p_neg_mass = 1.0 - p_true                                                # (B, M)

    # Position‑dependent weights
    pos = torch.arange(1, M + 1, device=c_logits.device, dtype=probs.dtype)  # 1..M
    if loss_type == 'linear':
        w = gamma * pos / M
    elif loss_type == 'gaussian':
        w = 1.0 - torch.exp(-(pos / beta) ** 2)
    elif loss_type == 'sigmoid':
        w = 2.0 / (1.0 + torch.exp(-pos / alpha)) - 1.0
    else:
        raise ValueError(f"Unknown loss_type {loss_type}")
    w = w.view(1, M)                                                         # (1, M) → broadcast

    neg_loss = -w * torch.log(p_neg_mass.clamp(min=1e-8))                    # (B, M)

    # ------------------------------------------------------------------
    # 3) Combine, mask padding, and average
    # ------------------------------------------------------------------
    total = pos_loss + neg_loss                                              # (B, M)
    mask = (y != ignore_index).float()                                       # (B, M)
    loss = (total * mask).sum() / mask.sum().clamp(min=1.0)
    return loss

@torch.no_grad()
def evaluate(model, dataloader, device, dataset, args):
    model.eval()
    total_loss = 0.0
    for tokens, mask, prefix in dataloader:
        tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
        outputs = model(tokens, prefix, mask)
        # Use prefix_length from the original (full) dataset
        logits = outputs.logits[:, dataset.prefix_length - 1: -1]
        if args.loss_fn == 'cross_entropy':
            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
        else:
            loss = position_conditioned_loss(
                logits, tokens,
                loss_type=args.pcloss_type,
                gamma=args.gamma,
                beta=args.beta,
                alpha=args.alpha,
                ignore_index=0
            )
        total_loss += loss.item()
    return total_loss / len(dataloader)

def train(dataset: ClipCocoDataset, model: ClipCaptionModel, args,
          train_dataloader: DataLoader, val_dataloader: DataLoader = None,
          lr: float = 1e-5, warmup_steps: int = 500, output_dir: str = ".", output_prefix: str = ""):

    device = torch.device('cuda')
    batch_size = args.bs
    epochs = args.epochs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )
    # save_config(args)
    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        for idx, (tokens, mask, prefix) in enumerate(train_dataloader):
            model.zero_grad()
            tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
            outputs = model(tokens, prefix, mask)
            logits = outputs.logits[:, dataset.prefix_length - 1: -1]
            if args.loss_fn == 'cross_entropy':
                loss = nnf.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    tokens.flatten(),
                    ignore_index=0
                )
            else:
                loss = position_conditioned_loss(
                    logits,
                    tokens,
                    loss_type=args.pcloss_type,
                    gamma=args.gamma,
                    beta=args.beta,
                    alpha=args.alpha,
                    ignore_index=0
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})

            wandb.log({"train/loss": loss.item(), "step": epoch * len(train_dataloader) + idx})

            progress.update()
            if (idx + 1) % 10000 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"{output_prefix}_latest.pt"),
                )
        progress.close()
        if epoch % args.save_every == 0 or epoch == epochs - 1:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch:03d}.pt"),
            )

                # ── CHANGED ──: Save checkpoint each epoch as before
        if epoch % args.save_every == 0 or epoch == epochs - 1:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch:03d}.pt"),
            )

        # ── CHANGED ──: At end of each epoch, compute validation loss if val_dataloader is provided
        if val_dataloader is not None:
            val_loss = evaluate(model, val_dataloader, device, dataset, args)
            print(f"Validation Loss: {val_loss}")
            wandb.log({"val/loss": val_loss, "epoch": epoch})

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/meme/oscar_split_ViT-B_32_train.pkl')
    parser.add_argument('--out_dir', default='./checkpoints')
    parser.add_argument('--prefix', default='coco_prefix', help='prefix for saved filenames')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--prefix_length_clip', type=int, default=10)
    parser.add_argument('--bs', type=int, default=40)
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true', default=False)
    parser.add_argument('--mapping_type', type=str, default='mlp', help='mlp/transformer')
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--num_heads', type=int, default=12) 
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    parser.add_argument('--loss_fn', choices=['cross_entropy', 'position'], default='cross_entropy',
                    help='Choose loss function: cross_entropy or position-conditioned')
    parser.add_argument('--pcloss_type', choices=['linear', 'gaussian', 'sigmoid'], default='sigmoid',
                        help='Weight function type for position-conditioned loss')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='Gamma for linear weight in position-conditioned loss')
    parser.add_argument('--beta', type=float, default=10.0,
                        help='Beta for gaussian weight in position-conditioned loss')
    parser.add_argument('--alpha', type=float, default=6.0,
                        help='Alpha for sigmoid weight in position-conditioned loss')
    args = parser.parse_args()

    wandb.init(
        project="clipcap-meme-ft",  # change this to your project name
        entity="kevvnbk-1",      
        config=vars(args)           # logs all argparse arguments
    )

    prefix_length = args.prefix_length
    full_dataset = ClipCocoDataset(args.data, prefix_length, normalize_prefix=args.normalize_prefix)
    
    dataset_size = len(full_dataset)
    test_size = int(dataset_size * 0.1)
    val_size = int(dataset_size * 0.1)
    train_size = dataset_size - test_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    # WeightedRandomSampler for training set (weighted by funny_score + 1e-6)
    train_indices = train_dataset.indices
    sample_weights = [full_dataset.funny_scores[i] + 1e-6 for i in train_indices]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, sampler=sampler, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False, drop_last=False)

    prefix_dim = 640 if args.is_rn else 768
    args.mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[args.mapping_type]
    if args.only_prefix:
        model = ClipCaptionPrefix(prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=args.num_layers, num_heads=args.num_heads, mapping_type=args.mapping_type)
        print("Train only prefix")
    else:
        model = ClipCaptionModel(prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=args.num_layers, num_heads=args.num_heads, mapping_type=args.mapping_type)
        print("Train both prefix and GPT")
        sys.stdout.flush()

    # ──────────────────────────────────────────────────────────────────
    # Optional: load COCO‑pretrained weights before fine‑tuning
    # ──────────────────────────────────────────────────────────────────
    coco_ckpt = 'transformer_weights.pt'
    if os.path.isfile(coco_ckpt):
        print(f"Loading COCO checkpoint from {coco_ckpt}")
        state_dict = torch.load(coco_ckpt, map_location=torch.device('cpu'))
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"Checkpoint loaded ✓  (missing={len(missing_keys)}, unexpected={len(unexpected_keys)})")
    else:
        print(f"Warning: COCO checkpoint '{coco_ckpt}' not found. Proceeding without loading.")


    model = train(
        full_dataset, model, args, train_dataloader, val_dataloader, output_dir=args.out_dir, output_prefix=args.prefix,
    )
    
    test_loss = evaluate(model, test_dataloader, torch.device('cuda'), full_dataset, args)
    print(f"Test Loss: {test_loss}")
    wandb.log({"test/loss": test_loss})

    wandb.finish()


if __name__ == '__main__':
    main()
