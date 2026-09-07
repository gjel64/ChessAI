import chess
import torch
import urllib.request
from tqdm import tqdm
import os
import zstandard
import io
import json
import math
import multiprocessing
import numpy as np

N_CASES = 64
N_PIECES = 13
N_META = 6


PIECE_TO_IDX = {
    '.': 0,
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12,
}


def normaliser(board):
    if board.turn == chess.BLACK:
        return board.mirror(), True
    return board, False


def encoder_cases(board):
    out = []
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        out.append(PIECE_TO_IDX['.' if piece is None else piece.symbol()])
    return torch.tensor(out, dtype=torch.long)


def encoder_meta(board):
    return torch.tensor([
        float(board.has_kingside_castling_rights(chess.WHITE)),    # petit roque
        float(board.has_queenside_castling_rights(chess.WHITE)),   # grand roque
        float(board.has_kingside_castling_rights(chess.BLACK)),    # petit roque
        float(board.has_queenside_castling_rights(chess.BLACK)),   # grand roque
        float(board.ep_square is not None),
        board.halfmove_clock / 100.0, # normalize
    ], dtype=torch.float)


def encoder_masque_legal(board):
    mask = torch.zeros(N_CASES, N_CASES, dtype=torch.bool)
    for move in board.legal_moves:
        mask[move.from_square, move.to_square] = True
    return mask


def encoder(board):
    board_n, miroite = normaliser(board)
    return {
        'squares': encoder_cases(board_n),
        'meta': encoder_meta(board_n),
        'legal_mask': encoder_masque_legal(board_n),
        'miroite': miroite,
    }


def decoder_coup(idx, miroite, promotion=None):
    depart, arrivee = divmod(int(idx), N_CASES)
    if miroite:
        depart = chess.square_mirror(depart)
        arrivee = chess.square_mirror(arrivee)
    return chess.Move(depart, arrivee, promotion=promotion)


def value_vers_blancs(value, board):
    return value if board.turn == chess.WHITE else -value


# --------------- ENCODAGE RAPIDE ---------------
#
# encoder() ci-dessus construit un echiquier miroir (board.mirror()) puis
# interroge les 64 cases une par une : ~113 us par position. Les fonctions qui
# suivent font le meme travail sans jamais copier l'echiquier -- le miroir est
# un simple XOR 56 sur le numero de case, et les couleurs se deduisent du trait.


def encoder_rapide(board):
    """(cases, meta, coups legaux) dans le repere du modele, sans board.mirror().

    Le modele voit toujours le joueur au trait comme "les blancs" : les cases
    sont retournees quand les noirs ont le trait (sq ^ 56), et une piece est
    codee 1..6 si elle est au trait, 7..12 sinon. Meme sortie que encoder().
    """
    noirs = board.turn == chess.BLACK
    flip = 56 if noirs else 0

    cases = np.zeros(N_CASES, dtype=np.uint8)
    for sq, piece in board.piece_map().items():
        cases[sq ^ flip] = piece.piece_type + (0 if piece.color == board.turn else 6)

    trait, adverse = board.turn, not board.turn
    meta = np.array([
        board.has_kingside_castling_rights(trait),
        board.has_queenside_castling_rights(trait),
        board.has_kingside_castling_rights(adverse),
        board.has_queenside_castling_rights(adverse),
        board.ep_square is not None,
        board.halfmove_clock / 100.0,
    ], dtype=np.float32)

    # unique : les quatre promotions d'un meme pion partagent leur (depart, arrivee)
    coups = np.unique(np.fromiter(
        ((m.from_square ^ flip) * N_CASES + (m.to_square ^ flip) for m in board.legal_moves),
        dtype=np.uint16))

    return cases, meta, coups


# --------------- CACHE BINAIRE ---------------
#
# Encoder la base a chaque entrainement coute ~11 min pour 17M positions, et la
# liste des FEN doit ensuite etre picklee vers chaque worker du DataLoader
# (~1.3 Go par worker sous spawn, le defaut de macOS). On encode donc une fois
# pour toutes vers des fichiers binaires, que l'entrainement lit en memmap :
# demarrage immediat, memoire quasi nulle, aucun worker.

VERSION_CACHE = 1

FICHIERS = {
    'cases':  ('cases.u8',    np.uint8),      # (N, 64)  la position
    'meta':   ('meta.f16',    np.float16),    # (N, 6)   roques, prise en passant, 50 coups
    'value':  ('value.f16',   np.float16),    # (N,)     cible value
    'index':  ('index.u16',   np.uint16),     # (N,)     cible policy
    'coups':  ('coups.u16',   np.uint16),     # (M,)     coups legaux, bout a bout
    'bornes': ('bornes.i64',  np.int64),      # (N+1,)   ou commence chaque position
    'fens':   ('fens.i64',    np.int64),      # (N+1,)   idem pour le texte des FEN
}


def _encoder_lignes(travail):
    """Encode un paquet de lignes de la base. Execute dans un processus fils."""
    lignes, mate_cp, echelle_cp = travail
    cases, meta, value, index, coups, tailles, fens = [], [], [], [], [], [], []

    for ligne in lignes:
        try:
            entree = json.loads(ligne)
        except json.JSONDecodeError:
            continue                    # derniere ligne coupee : base tronquee

        pv = best_pv(entree)
        uci = goal_policy(pv)
        if uci is None:                 # pas de variante -> pas de cible policy
            continue

        board = chess.Board(entree['fen'])
        try:
            idx = index_coup(board, uci, board.turn == chess.BLACK)
        except ValueError:
            continue                    # position Chess960 : son roque n'a pas de sens ici

        c, m, legaux = encoder_rapide(board)
        cases.append(c)
        meta.append(m)
        value.append(goal_value(pv, board.turn == chess.WHITE, mate_cp, echelle_cp))
        index.append(idx)
        coups.append(legaux)
        tailles.append(len(legaux))
        fens.append(entree['fen'])

    if not cases:
        return None

    return (np.stack(cases), np.stack(meta).astype(np.float16),
            np.array(value, dtype=np.float16), np.array(index, dtype=np.uint16),
            np.concatenate(coups), np.array(tailles, dtype=np.int64),
            ('\n'.join(fens) + '\n').encode())


def _paquets(chemin, n_lignes):
    """Decoupe la base en paquets de lignes, sans rien decoder."""
    with open_base(chemin) as f:
        paquet = []
        for ligne in f:
            paquet.append(ligne)
            if len(paquet) >= n_lignes:
                yield paquet
                paquet = []
        if paquet:
            yield paquet


def preparer_cache(chemin, dossier, n_max, mate_cp, echelle_cp, n_procs=None, paquet=20000):
    """Encode la base dans dossier/, une fois pour toutes. Renvoie le nombre de positions.

    Le cache deja present est reutilise s'il contient assez de positions et a ete
    produit avec les memes parametres de cible.
    """
    params = {'version': VERSION_CACHE, 'mate_cp': mate_cp, 'echelle_cp': echelle_cp,
              'octets_source': os.path.getsize(chemin)}
    infos = os.path.join(dossier, 'cache.json')

    if os.path.exists(infos):
        with open(infos) as f:
            deja = json.load(f)
        # 'epuise' : la base entiere y est deja passee, en redemander plus est vain
        if (all(deja.get(k) == v for k, v in params.items())
                and (deja['n'] >= n_max or deja.get('epuise'))):
            print(f'cache : {deja["n"]} positions deja encodees dans {dossier}/')
            return deja['n']

    os.makedirs(dossier, exist_ok=True)
    sorties = {nom: open(os.path.join(dossier, fichier), 'wb')
               for nom, (fichier, _) in FICHIERS.items()}
    textes = open(os.path.join(dossier, 'fens.txt'), 'wb')

    n = m = octets = 0
    sorties['bornes'].write(np.zeros(1, dtype=np.int64).tobytes())
    sorties['fens'].write(np.zeros(1, dtype=np.int64).tobytes())

    n_procs = n_procs or max(1, (os.cpu_count() or 2) - 1)
    bar = tqdm(total=n_max, desc='encodage de la base')

    with multiprocessing.Pool(n_procs) as pool:
        travaux = ((p, mate_cp, echelle_cp) for p in _paquets(chemin, paquet))
        for resultat in pool.imap(_encoder_lignes, travaux):
            if resultat is None:
                continue
            cases, meta, value, index, coups, tailles, fens = resultat

            garde = min(len(cases), n_max - n)          # ne pas depasser n_max
            if garde < len(cases):
                cases, meta, value, index = cases[:garde], meta[:garde], value[:garde], index[:garde]
                coups = coups[:int(tailles[:garde].sum())]
                fens = b'\n'.join(fens.split(b'\n')[:garde]) + b'\n'
                tailles = tailles[:garde]

            sorties['cases'].write(cases.tobytes())
            sorties['meta'].write(meta.tobytes())
            sorties['value'].write(value.tobytes())
            sorties['index'].write(index.tobytes())
            sorties['coups'].write(coups.tobytes())
            sorties['bornes'].write((m + np.cumsum(tailles)).tobytes())

            longueurs = np.frombuffer(fens, dtype=np.uint8) == 10        # fins de ligne
            sorties['fens'].write((octets + np.flatnonzero(longueurs) + 1).tobytes())
            textes.write(fens)

            n += garde
            m += int(tailles.sum())
            octets += len(fens)
            bar.update(garde)

            if n >= n_max:
                break

    bar.close()
    for f in sorties.values():
        f.close()
    textes.close()

    with open(infos, 'w') as f:
        json.dump({**params, 'n': n, 'm': m, 'epuise': n < n_max}, f)

    total = sum(os.path.getsize(os.path.join(dossier, nom))
                for nom in os.listdir(dossier))
    print(f'cache : {n} positions encodees dans {dossier}/ ({total / 1e9:.2f} Go)')
    return n


class Cache:
    """Lecture du cache en memmap : rien n'est charge en memoire."""

    def __init__(self, dossier, n=None):
        with open(os.path.join(dossier, 'cache.json')) as f:
            self.infos = json.load(f)
        self.n = min(n or self.infos['n'], self.infos['n'])
        self.dossier = dossier

        def lire(nom, forme):
            fichier, dtype = FICHIERS[nom]
            return np.memmap(os.path.join(dossier, fichier), dtype=dtype, mode='r', shape=forme)

        self.cases = lire('cases', (self.infos['n'], N_CASES))
        self.meta = lire('meta', (self.infos['n'], N_META))
        self.value = lire('value', (self.infos['n'],))
        self.index = lire('index', (self.infos['n'],))
        self.coups = lire('coups', (self.infos['m'],))
        self.bornes = lire('bornes', (self.infos['n'] + 1,))
        self.fens_bornes = lire('fens', (self.infos['n'] + 1,))
        self.fens_texte = np.memmap(os.path.join(dossier, 'fens.txt'), dtype=np.uint8, mode='r')

    def __len__(self):
        return self.n

    def fen(self, i):
        debut, fin = self.fens_bornes[i], self.fens_bornes[i + 1]
        return bytes(self.fens_texte[debut:fin]).decode().strip()

    def lot(self, indices):
        """Un batch pret pour le modele : (cases, meta, masque legal, value, index)."""
        cases = torch.from_numpy(self.cases[indices].astype(np.int64))
        meta = torch.from_numpy(self.meta[indices].astype(np.float32))
        value = torch.from_numpy(self.value[indices].astype(np.float32)).unsqueeze(1)
        index = torch.from_numpy(self.index[indices].astype(np.int64))

        # Les coups legaux sont ranges bout a bout : on reconstitue le masque
        # (B, 64, 64) d'un seul coup, sans boucle Python.
        debuts = self.bornes[indices]
        tailles = self.bornes[np.asarray(indices) + 1] - debuts
        depart = np.repeat(np.cumsum(tailles) - tailles, tailles)
        plats = np.arange(int(tailles.sum())) - depart + np.repeat(debuts, tailles)

        masque = np.zeros((len(indices), N_CASES * N_CASES), dtype=bool)
        masque[np.repeat(np.arange(len(indices)), tailles), self.coups[plats]] = True

        return (cases, meta, torch.from_numpy(masque).view(-1, N_CASES, N_CASES),
                value, index)


class Chargeur:
    """Itere le cache par batches. Remplace DataLoader : ni worker, ni pickle."""

    def __init__(self, cache, indices, batch_size, melanger=False, graine=None):
        self.cache = cache
        self.indices = np.asarray(indices)
        self.batch_size = batch_size
        self.melanger = melanger
        self.rng = np.random.default_rng(graine)

    def __len__(self):
        # a l'entrainement on jette le reste, en validation on le garde
        if self.melanger:
            return len(self.indices) // self.batch_size
        return -(-len(self.indices) // self.batch_size)

    def __iter__(self):
        ordre = self.rng.permutation(self.indices) if self.melanger else self.indices
        for debut in range(0, len(self) * self.batch_size, self.batch_size):
            lot = np.sort(ordre[debut:debut + self.batch_size])   # lecture memmap ordonnee
            yield self.cache.lot(lot)


def download_data(chemin, octets, url, n_pos):
    """Telecharge le debut de la base Lichess, ou complete ce qui est deja la.

    Le fichier complet fait 21 Go : on ne prend que les premiers octets, via une
    requete HTTP Range. Un .zst tronque se decompresse jusqu'a la coupure, et
    reprendre depuis la taille du fichier suffit a en telecharger davantage.
    """
    already = os.path.getsize(chemin) if os.path.exists(chemin) else 0
    if already >= octets:
        print(f'{chemin} : {already / 1e9:.2f} Go already here, enough for {n_pos} positions')
        return

    print(f'download {(octets - already) / 1e9:.2f} Go from {url}')
    requete = urllib.request.Request(url, headers={'Range': f'bytes={already}-{octets - 1}'})

    with urllib.request.urlopen(requete) as reponse, open(chemin, 'ab') as out:
        bar = tqdm(total=octets - already, unit='o', unit_scale=True)
        while bloc := reponse.read(1 << 20):
            out.write(bloc)
            bar.update(len(bloc))
        bar.close()

def open_base(chemin):
    """La base est distribuee compressee (.zst), mais un .jsonl brut marche aussi."""
    if not chemin.endswith('.zst'):
        return open(chemin, 'r')

    flux = zstandard.ZstdDecompressor().stream_reader(open(chemin, 'rb'))
    return io.TextIOWrapper(flux, encoding='utf-8')


def read_data(chemin):
    """Parcourt la base position par position, sans tout charger en memoire."""
    with open_base(chemin) as f:
        try:
            for ligne in f:
                yield json.loads(ligne)
        except (json.JSONDecodeError, zstandard.ZstdError):
            return      # derniere ligne coupee : la base est volontairement tronquee



def best_pv(entree):
    """L'eval la plus profonde de la position, ligne principale."""
    best = max(entree['evals'], key=lambda e: e['depth'])
    return best['pvs'][0]

def goal_value(pv, trait_blanc, mate_cp, echelle_cp):
    """Eval Stockfish (point de vue blancs) -> cible dans [-1, 1] pour le joueur au trait."""
    if pv.get('mate') is not None:
        cp = mate_cp if pv['mate'] > 0 else -mate_cp
    else:
        cp = float(pv['cp'])

    if not trait_blanc:
        cp = -cp

    return math.tanh(cp / echelle_cp)

def goal_policy(pv):
    """Premier coup de la variante principale = meilleur coup selon Stockfish."""
    line = pv.get('line')
    return line.split(' ')[0] if line else None


def index_coup(board, uci, miroite):
    """Coup UCI -> index dans la policy (64*64). Inverse de decoder_coup de data.py.

    parse_uci normalise le roque : Lichess le note "roi prend tour" (e1h1),
    la ou python-chess attend e1g1. Sans ca la cible tombe sur une case masquee.
    """
    coup = board.parse_uci(uci)
    start, end = coup.from_square, coup.to_square

    if miroite:
        start = chess.square_mirror(start)
        end = chess.square_mirror(end)

    return start * N_CASES + end


def load_positions(chemin, n_max, mate_cp, echelle_cp):
    """Renvoie une liste de (fen, goal_value, coup_uci, index_policy)."""
    positions = []
    bar = tqdm(total=n_max, desc='lecture de la base')

    for input in read_data(chemin):
        if len(positions) >= n_max:
            break

        pv = best_pv(input)
        coup = goal_policy(pv)
        if coup is None:                # pas de variante -> pas de cible policy
            continue

        fen = input['fen']
        board = chess.Board(fen)

        try:
            index = index_coup(board, coup, board.turn == chess.BLACK)
        except ValueError:
            continue        # position Chess960 : son roque n'a pas de sens ici

        positions.append((fen, goal_value(pv, board.turn == chess.WHITE, mate_cp, echelle_cp), coup, index))
        bar.update(1)

    bar.close()
    return positions