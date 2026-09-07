"""Entrainement supervise sur les evaluations Stockfish de Lichess.

Base : https://database.lichess.org/#evals  ->  lichess_db_eval.jsonl.zst
Une ligne = une position :
    {"fen": "...", "evals": [{"pvs": [{"cp": 311, "line": "e2e4 e7e5 ..."}], "depth": 36}, ...]}

Deux cibles sont extraites de chaque position :
  - value  : l'eval Stockfish ramenee dans [-1, 1] (comme le tanh de la value head) ;
  - policy : le premier coup de la variante principale, soit le meilleur coup de Stockfish.

Les deux sont exprimees du point de vue du joueur au trait, comme l'encodage de data.py.
"""


import json
import math
import random

import chess
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from data import (decoder_coup, N_PIECES, N_CASES, download_data,
                  preparer_cache, Cache, Chargeur)
from model import Transformer


URL_DATA = 'https://database.lichess.org/lichess_db_eval.jsonl.zst'
DATA_PATH = 'lichess_db_eval.jsonl.zst'
CACHE_PATH = 'cache'      # base encodee une fois pour toutes, voir data.py
MODEL_PATH = 'model.pt'

N_POSITIONS = 1_024 * 65_536    # nombre de positions lues dans la base = 65M
VAL_PART = 0.05

# La base complete fait 21 Go, pour ~100 octets compresses par position.
# On n'en telecharge que le debut : de quoi couvrir N_POSITIONS, avec de la marge.
OCTETS_A_TELECHARGER = N_POSITIONS * 150

ECHELLE_CP = 400.0        # 400 centipions -> tanh(1) ~ 0.76
MATE_CP = 2000.0          # un mat vaut ~ +/- 1
POIDS_POLICY = 1.0        # poids de la perte policy face a la perte value

BATCH_SIZE = 1024
LR = 4e-3 # found 

PART_WARMUP = 0.02
LR_MIN_PART = 0.05

GRAD_CLIP = 1.0

AMP = True
AMP_DTYPE = torch.bfloat16

EMB_DIM = 128
N_HEADS = 8
N_BLOCKS = 6
DROPOUT = 0.0

SEED = 7

# --------------- LOSS ---------------

def compute_loss(model, batch, device, with_accuracy=True):
    """Renvoie (perte totale, perte value, perte policy, precision policy)."""
    squares, meta, legal_mask, goal_v, goal_p = [t.to(device) for t in batch]

    with torch.autocast(device_type=device.type, dtype=AMP_DTYPE,
                        enabled=AMP and device.type != 'cpu'):
        policy, value = model(squares, meta, legal_mask)

    policy, value = policy.float(), value.float()

    loss_v = F.mse_loss(value, goal_v)
    loss_p = F.cross_entropy(policy, goal_p)
    loss = loss_v + POIDS_POLICY * loss_p

    if not with_accuracy:
        return loss, loss_v, loss_p

    accuracy = (policy.argmax(dim=1) == goal_p).float().mean()

    return loss, loss_v, loss_p, accuracy


# --------------- TRAINING ---------------

@torch.no_grad()
def eval(model, loader, device):
    model.eval()
    total, n = torch.zeros(3), 0

    for lot in loader:
        size = len(lot[0])
        _, loss_v, loss_p, accuracy = compute_loss(model, lot, device)
        total += torch.tensor([loss_v.item(), loss_p.item(), accuracy.item()]) * size
        n += size

    return (total / max(n, 1)).tolist()


def train():
    torch.manual_seed(SEED)
    random.seed(SEED)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'device : {device}')

    download_data(DATA_PATH, OCTETS_A_TELECHARGER, URL_DATA, N_POSITIONS)

    preparer_cache(DATA_PATH, CACHE_PATH, N_POSITIONS, MATE_CP, ECHELLE_CP)
    cache = Cache(CACHE_PATH, n=N_POSITIONS)

    melange = np.random.default_rng(SEED).permutation(len(cache))
    n_val = int(len(cache) * VAL_PART)
    idx_val, idx_train = melange[:n_val], melange[n_val:]
    print(f'{len(idx_train)} positions train / {len(idx_val)} val')

    loader_train = Chargeur(cache, idx_train, BATCH_SIZE, melanger=True, graine=SEED)
    loader_val = Chargeur(cache, idx_val, BATCH_SIZE)

    model = Transformer(
        vocab_size=N_PIECES,
        emb_dim=EMB_DIM,
        n_heads=N_HEADS,
        context_len=N_CASES,
        n_block=N_BLOCKS,
        dropout=DROPOUT,
    ).to(device)


    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05,fused=True)

    n_steps = len(loader_train)
    n_warmup = max(1, int(n_steps * PART_WARMUP))

    def facteur_lr(step):
        if step < n_warmup:
            return (step + 1) / n_warmup
        avance = (step - n_warmup) / max(1, n_steps - n_warmup)
        cosinus = 0.5 * (1 + math.cos(math.pi * min(avance, 1.0)))
        return LR_MIN_PART + (1 - LR_MIN_PART) * cosinus

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, facteur_lr)
    print(f'{n_steps} steps ({n_warmup} warmup), LR {LR:.1e} -> {LR * LR_MIN_PART:.1e}')

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{(trainable_params / 1_000_000):.2}M parameters")

    model.train()
    log_loss = []
    log = {
        "loss" : [],
        "lr" : []
    }

    for i, batch in enumerate(tqdm(loader_train)):

        loss, loss_v, loss_p = compute_loss(model, batch, device, with_accuracy=False)
    
        optim.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optim.step()
        scheduler.step()

        if (i % 100 == 0):
            log_loss.append(loss.item())
            log["loss"].append(sum(log_loss[-100:]) / min(len(log_loss), 100))
            log['lr'].append(scheduler.get_last_lr()[0])
        

        if (i % 1000 == 0):
            print(f'step {i} : {log["loss"][-1]:.4f}  (lr {scheduler.get_last_lr()[0]:.2e})')
            torch.save(model.state_dict(), MODEL_PATH)
            with open("log_loss.txt", "w") as f:
                f.write(json.dumps(log))
        

    loss_v, loss_p, accuracy = eval(model, loader_val, device)
    

    print(f'validation : value {loss_v:.4f} '
            f'(erreur ~{math.sqrt(loss_v):.3f}) | policy {loss_p:.4f} | '
            f'meilleur coup trouve {accuracy:.1%}')

    torch.save(model.state_dict(), MODEL_PATH)
    print(f'model sauvegarde dans {MODEL_PATH}')

    with open("log_loss.txt", "w") as f:
        f.write(json.dumps(log))
    
    evaluate(model, cache, idx_val, device)


@torch.no_grad()
def evaluate(model, cache, idx_val, device, n=10):
    """Quelques predictions du modele face a Stockfish."""
    model.eval()
    print('\nmodele vs Stockfish (point de vue du joueur au trait)')

    for i in idx_val[:n]:
        squares, meta, legal_mask, value_sf, index_sf = cache.lot([i])
        policy, value = model(squares.to(device), meta.to(device), legal_mask.to(device))

        fen = cache.fen(i)
        noirs = chess.Board(fen).turn == chess.BLACK
        coup = decoder_coup(policy[0].argmax().item(), noirs)
        coup_sf = decoder_coup(int(index_sf[0]), noirs)

        print(f'value {value.item():+.3f} / {value_sf.item():+.3f}  |  '
              f'coup {coup.uci()} / {coup_sf.uci()}  |  {fen}')


if __name__ == '__main__':
    train()
