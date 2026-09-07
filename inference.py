"""Jouer une partie contre le modele, dans une fenetre pygame.

    python inference.py                        # tu joues les blancs
    python inference.py --couleur noir
    python inference.py --mode policy --temp 0.5
    python inference.py --fen "8/8/8/4k3/8/8/4P3/4K3 w - - 0 1"

Souris : clic (ou glisser) sur une piece puis sur sa case d'arrivee.
Clavier : n nouvelle partie, u annuler, f retourner l'echiquier,
          h indice, m changer de mode, echap quitter.

Trois facons de choisir le coup (--mode). Mesure sur 800 positions de la base :
part des coups identiques a Stockfish, et part des coups qui laissent une piece
en prise (une reprise simple perd au moins un cavalier net).

  recherche : defaut. MCTS PUCT de --sims simulations, guide par la policy
              (prior) et la value (evaluation des feuilles). Remplace l'ancien
              depth-2 + LAMBDA_PRIOR : la profondeur n'est plus fixe, l'arbre
              creuse la ou la policy et la value ne sont pas d'accord. Les
              chiffres ci-dessous sont ceux de l'ANCIENNE recherche, a remesurer.
                                                        42.9% / 17.9% (obsolete)
  policy    : le coup le plus probable selon la tete policy, sans recherche.
                                                        40.1% / 20.4%
  value     : chaque coup legal est joue et note par la tete value. Plus lent et
              plus faible : la value se trompe de ~100 cp, la policy est bien
              plus fiable qu'elle.                      25.5% /   --
"""

import argparse
import math
import os
import random
import threading

import chess
import pygame
import torch
from torch.nn import functional as F

from data import encoder, N_CASES, N_PIECES
from model import Transformer


MODEL_PATH = 'model.pt'

# Valeurs de repli seulement : l'architecture est deduite du checkpoint lui-meme
# (voir architecture_du_checkpoint), donc plus besoin de la recopier a la main
# depuis supervised-learning.py a chaque changement.
EMB_DIM = 128
N_HEADS = 8
N_BLOCKS = 6

ECHELLE_CP = 400.0      # inverse de goal_value : tanh(cp / 400)

# REMPLACEMENT DE recherche_2 PAR UN MCTS : LAMBDA_PRIOR disparait. Melanger a
# la main le score de recherche et log(proba policy) etait un bricolage ; dans
# PUCT le prior de la policy est deja dans la formule de selection, et son poids
# decroit tout seul a mesure qu'un coup est visite.
C_PUCT = 1.5            # exploration : plus haut = plus fidele a la policy
N_SIMULATIONS = 400     # simulations par coup (--sims)


# =============== MODELE ===============

def choisir_device(nom=None):
    if nom:
        return torch.device(nom)
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def architecture_du_checkpoint(state):
    """Deduit (emb_dim, n_heads, n_block) des tenseurs du state_dict.

    Le checkpoint ne stocke pas les hyperparametres, mais leurs traces sont dans
    les formes : emb.weight est (vocab, emb_dim), biais_attention est
    (1, n_heads, 64, 64), et les blocs sont numerotes blocks.<i>.*.
    """
    emb_dim = state['emb.weight'].shape[1]

    if 'biais_attention' in state:
        n_heads = state['biais_attention'].shape[1]
    else:
        n_heads = N_HEADS       # checkpoint anterieur au biais : on ne peut pas deduire

    indices = [int(k.split('.')[1]) for k in state if k.startswith('blocks.')]
    n_block = max(indices) + 1 if indices else N_BLOCKS

    return emb_dim, n_heads, n_block


def charger_modele(chemin, device):
    state = torch.load(chemin, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']

    emb_dim, n_heads, n_block = architecture_du_checkpoint(state)
    print(f'{chemin} : emb_dim {emb_dim}, {n_heads} tetes, {n_block} blocs')

    model = Transformer(
        vocab_size=N_PIECES,
        emb_dim=emb_dim,
        n_heads=n_heads,
        context_len=N_CASES,
        n_block=n_block,
        dropout=0.0,        # pas de dropout en inference
    ).to(device)

    # AJOUT DU BIAIS D'ATTENTION : les checkpoints entraines avant le biais
    # (model_.pt, model1.pt) n'ont pas la cle biais_attention. Comme elle est
    # initialisee a zero et que le biais est alors un no-op, on peut les charger
    # tels quels : le modele se comporte exactement comme avant.
    manquantes, inattendues = model.load_state_dict(state, strict=False)
    manquantes = [k for k in manquantes if k != 'biais_attention']
    if manquantes or inattendues:
        raise RuntimeError(f'checkpoint incompatible : manquant {manquantes}, '
                           f'inattendu {list(inattendues)}')
    if 'biais_attention' not in state:
        print(f'{chemin} : pas de biais d\'attention (checkpoint anterieur), '
              'laisse a zero')

    model.eval()
    return model


def encoder_lot(boards, device):
    """Encode une liste de positions en un seul lot (B, ...)."""
    encodes = [encoder(b) for b in boards]
    return (
        torch.stack([e['squares'] for e in encodes]).to(device),
        torch.stack([e['meta'] for e in encodes]).to(device),
        torch.stack([e['legal_mask'] for e in encodes]).to(device),
    )


def index_policy(board, coup):
    """(depart, arrivee) du coup dans le repere du modele -> index 0..64*64-1.

    encoder() retourne l'echiquier quand les noirs ont le trait : les index de
    la policy vivent dans ce repere miroir, pas dans celui de la partie.
    """
    depart, arrivee = coup.from_square, coup.to_square
    if board.turn == chess.BLACK:
        depart = chess.square_mirror(depart)
        arrivee = chess.square_mirror(arrivee)
    return depart * N_CASES + arrivee


@torch.no_grad()
def probas_coups(model, board, device):
    """Coups legaux tries par probabilite decroissante : [(coup, proba), ...].

    Les sous-promotions partagent l'index (depart, arrivee) de la promotion en
    dame : elles recoivent donc le meme logit, et la dame est gardee en tete.
    """
    squares, meta, legal_mask = encoder_lot([board], device)
    policy, value = model(squares, meta, legal_mask)

    logits = policy[0].float().cpu()
    coups = list(board.legal_moves)
    scores = torch.tensor([logits[index_policy(board, c)] for c in coups])
    probas = F.softmax(scores, dim=0)

    ordre = sorted(zip(coups, probas.tolist()),
                   key=lambda cp: (-cp[1], cp[0].promotion != chess.QUEEN))
    return ordre, value.item()


@torch.no_grad()
def evaluer_enfants(model, board, coups, device):
    """Note chaque coup par la value de la position obtenue, vue du joueur qui joue.

    La value head parle du joueur au trait : apres notre coup c'est l'adversaire,
    d'ou le signe oppose. Mats et nulles sont traites a part, le modele n'ayant
    jamais vu de position terminale.
    """
    scores = [None] * len(coups)
    a_evaluer, enfants = [], []

    for i, coup in enumerate(coups):
        board.push(coup)
        if board.is_checkmate():
            scores[i] = 1.0                     # mat pour nous
        elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            scores[i] = 0.0
        else:
            a_evaluer.append(i)
            enfants.append(board.copy(stack=False))
        board.pop()

    if a_evaluer:
        squares, meta, legal_mask = encoder_lot(enfants, device)
        _, values = model(squares, meta, legal_mask)
        for i, v in zip(a_evaluer, values.flatten().tolist()):
            scores[i] = -v

    return scores


# REMPLACEMENT DE recherche_2 PAR UN MCTS PUCT (AlphaZero, sans le self-play).
# recherche_2 explorait les topk coups a profondeur 2 fixe, puis melangeait son
# score avec la policy via LAMBDA_PRIOR. Trois limites : la profondeur ne
# s'adaptait pas (un echange force sur 6 demi-coups etait invisible), le budget
# etait reparti uniformement entre les topk coups, et le melange final etait
# arbitraire. PUCT regle les trois : il descend la ou Q + U est maximal, donc
# creuse les lignes prometteuses ou incertaines et abandonne les autres.


class Noeud:
    """Un noeud de l'arbre. Q est toujours vu du joueur au trait DANS ce noeud."""

    __slots__ = ('prior', 'visites', 'somme', 'enfants', 'terminal')

    def __init__(self, prior):
        self.prior = prior
        self.visites = 0
        self.somme = 0.0
        self.enfants = None     # dict coup -> Noeud ; None tant que non etendu
        self.terminal = None    # score de fin de partie, sinon None

    @property
    def q(self):
        return self.somme / self.visites if self.visites else 0.0


def fin_de_partie(board):
    """Score d'une position finie, vu du joueur au trait. Sinon None.

    Remplace score_terminal, qui n'avait que recherche_2 pour appelant : son
    can_claim_draw() rejoue tout l'historique a chaque appel, ce qui en ferait
    ici le poste le plus cher de l'arbre. On lui prefere la convention des
    moteurs : une seule repetition dans la recherche vaut nulle.
    """
    if board.is_checkmate():
        return -1.0                             # au trait et mat : perdu
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0
    if board.halfmove_clock >= 100 or board.is_repetition(2):
        return 0.0
    return None


@torch.no_grad()
def etendre(model, noeud, board, device):
    """Evalue la feuille et lui cree ses enfants. Renvoie la value du joueur au trait.

    Les quatre promotions d'un pion partagent leur index de policy, donc leur
    prior. Contrairement a probas_coups (qui garde la dame en tete et jette le
    reste), on les garde toutes les quatre comme enfants distincts : la value
    peut alors decouvrir qu'une sous-promotion est meilleure, ce dont l'ancienne
    recherche etait par construction incapable.
    """
    terminal = fin_de_partie(board)
    if terminal is not None:
        noeud.terminal = terminal
        return terminal

    ordre, value = probas_coups(model, board, device)
    noeud.enfants = {coup: Noeud(proba) for coup, proba in ordre}
    return value


def selection_puct(noeud, c_puct):
    """L'enfant qui maximise Q + U. Renvoie (coup, enfant)."""
    racine_n = math.sqrt(max(noeud.visites, 1))
    meilleur, meilleur_score = None, -math.inf

    for coup, enfant in noeud.enfants.items():
        # enfant.q est vu du joueur au trait dans l'enfant, soit l'adversaire :
        # de notre point de vue c'est -q. Un enfant jamais visite vaut 0 (nulle),
        # c'est seulement U qui le fait explorer.
        q = -enfant.q if enfant.visites else 0.0
        u = c_puct * enfant.prior * racine_n / (1 + enfant.visites)
        if q + u > meilleur_score:
            meilleur_score, meilleur = q + u, (coup, enfant)

    return meilleur


@torch.no_grad()
def mcts(model, board, device, n_sims=N_SIMULATIONS, c_puct=C_PUCT):
    """Construit l'arbre depuis board et renvoie sa racine, deja etendue.

    board est restaure a l'identique en sortie (push/pop symetriques), aucune
    copie de position n'est faite pendant la descente.
    """
    racine = Noeud(1.0)
    etendre(model, racine, board, device)
    if not racine.enfants:                      # mat ou pat : rien a chercher
        return racine

    for _ in range(n_sims):
        noeud, chemin, profondeur = racine, [racine], 0

        # SELECTION : descendre jusqu'a une feuille (non etendue ou terminale)
        while noeud.enfants is not None and noeud.terminal is None:
            coup, noeud = selection_puct(noeud, c_puct)
            board.push(coup)
            chemin.append(noeud)
            profondeur += 1

        # EVALUATION : la feuille est etendue, ou son score de fin est reutilise
        value = noeud.terminal if noeud.terminal is not None \
            else etendre(model, noeud, board, device)

        # RETROPROPAGATION : chaque demi-coup remonte change de camp, donc de signe
        for n in reversed(chemin):
            n.visites += 1
            n.somme += value
            value = -value

        for _ in range(profondeur):
            board.pop()

    return racine


def choisir_coup(model, board, device, mode='recherche', n_sims=N_SIMULATIONS,
                 temperature=0.0):
    """Renvoie (coup, value de la position courante, classement affichable).

    classement : [(coup, poids, score), ...], meilleur en tete. Le poids est la
    proba de la policy en mode policy/value ; en mode recherche c'est la part
    des visites du MCTS, c'est-a-dire la policy amelioree par la recherche.

    criteres vit toujours en log : softmax(criteres / T) donne alors p^(1/T) en
    mode policy et N^(1/T) en mode recherche, l'echantillonnage d'AlphaZero.
    """
    ordre, value = probas_coups(model, board, device)
    if not ordre:                   # mat ou pat : rien a choisir
        return None, value, []

    if mode == 'policy':
        classement = [(c, p, None) for c, p in ordre]
        criteres = torch.tensor([p for _, p in ordre]).clamp(min=1e-9).log()

    elif mode == 'value':
        scores = evaluer_enfants(model, board, [c for c, _ in ordre], device)
        classement = [(c, p, s) for (c, p), s in zip(ordre, scores)]
        # les scores vivent dans [-1, 1] : *5 les ramene a des logits comparables
        criteres = torch.tensor(scores) * 5.0

    else:
        # REMPLACEMENT DE recherche_2 PAR UN MCTS : le coup joue est le plus
        # VISITE, pas celui de meilleur Q. Un Q eleve vu deux fois est du bruit ;
        # les visites, elles, integrent la confiance de l'arbre.
        racine = mcts(model, board, device, n_sims)
        enfants = list(racine.enfants.items())
        total = max(sum(e.visites for _, e in enfants), 1)
        # -e.q : le Q de l'enfant est vu de l'adversaire, on l'affiche du notre
        classement = [(c, e.visites / total, -e.q) for c, e in enfants]
        criteres = torch.tensor([float(e.visites) for _, e in enfants]).clamp(min=1e-9).log()

    ordre_tri = criteres.argsort(descending=True, stable=True).tolist()
    classement = [classement[i] for i in ordre_tri]
    criteres = criteres[ordre_tri]

    if temperature > 0:
        poids = F.softmax(criteres / temperature, dim=0)
        coup = classement[torch.multinomial(poids, 1).item()][0]
    else:
        coup = classement[0][0]

    return coup, value, classement


def en_centipions(value):
    """value dans [-1, 1] -> centipions, l'inverse de tanh(cp / 400)."""
    v = max(min(value, 0.999), -0.999)
    return ECHELLE_CP * torch.atanh(torch.tensor(v)).item()


# =============== INTERFACE ===============

CASE = 82
PLATEAU = 8 * CASE
MARGE = 22
BARRE_L = 20            # largeur de la barre d'evaluation
PANNEAU_L = 320
LARGEUR = MARGE + BARRE_L + 12 + PLATEAU + 16 + PANNEAU_L + MARGE
HAUTEUR = MARGE + PLATEAU + MARGE

FOND = (33, 31, 29)
CLAIR = (238, 216, 185)
FONCE = (170, 124, 88)
SELECTION = (246, 216, 96)
DERNIER = (246, 216, 96)
ECHEC = (214, 94, 78)
TEXTE = (232, 228, 222)
TEXTE_GRIS = (150, 145, 138)
ACCENT = (126, 176, 118)
BOUTON = (56, 53, 49)
BOUTON_SURVOL = (74, 70, 65)

# Glyphes pleins pour les deux couleurs : les blancs sont peints en clair avec
# un contour sombre, bien plus lisible que les glyphes creux de la police.
GLYPHE = {chess.PAWN: '♟', chess.KNIGHT: '♞', chess.BISHOP: '♝',
          chess.ROOK: '♜', chess.QUEEN: '♛', chess.KING: '♚'}

POLICES_PIECES = [
    '/System/Library/Fonts/Apple Symbols.ttf',
    '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
]

PROMOTIONS = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]


def police_pieces(taille):
    """Premiere police disponible qui dessine vraiment les pieces d'echecs."""
    chemins = list(POLICES_PIECES)

    try:                                    # matplotlib embarque DejaVu Sans
        import matplotlib
        chemins.append(os.path.join(os.path.dirname(matplotlib.__file__),
                                    'mpl-data/fonts/ttf/DejaVuSans.ttf'))
    except ImportError:
        pass

    for chemin in chemins:
        if not os.path.exists(chemin):
            continue
        police = pygame.font.Font(chemin, taille)
        rendu = police.render(GLYPHE[chess.KNIGHT], True, (255, 255, 255))
        if pygame.mask.from_surface(rendu).count() > 0:      # pas un carre vide
            return police

    return pygame.font.SysFont(None, taille)


class Jeu:
    def __init__(self, args):
        self.args = args
        self.device = choisir_device(args.device)
        self.model = charger_modele(args.model, self.device)

        self.mode = args.mode
        self.board = chess.Board(args.fen) if args.fen else chess.Board()
        self.depart_fen = self.board.fen()

        if args.couleur == 'hasard':
            self.joueur = random.choice([chess.WHITE, chess.BLACK])
        else:
            self.joueur = chess.WHITE if args.couleur == 'blanc' else chess.BLACK
        self.bas = self.joueur           # couleur affichee en bas

        self.historique = []             # SAN des coups joues
        self.selection = None            # case de depart choisie
        self.glisse = None               # (case, piece) en cours de glisser
        self.souris = (0, 0)
        self.promotion = None            # (depart, arrivee) en attente de choix
        self.message = ''

        # Resultats du modele, remplis par le thread de calcul.
        self.classement = []
        self.value = 0.0
        self.value_affichee = 0.0        # barre d'eval, lissee
        self.indice = None
        self.classement_fen = None       # position a laquelle se rapporte classement
        self.calcul = None               # thread en cours
        self.attente = None              # tache demandee, pas encore demarree
        self.a_jouer = None              # coup renvoye par le thread
        self.reflechit = False

        pygame.init()
        pygame.display.set_caption('ChessAI')
        self.ecran = pygame.display.set_mode((LARGEUR, HAUTEUR))
        self.horloge = pygame.time.Clock()

        self.f_piece = police_pieces(int(CASE * 0.78))
        self.f_titre = pygame.font.SysFont('helveticaneue,helvetica,arial', 26, bold=True)
        self.f_texte = pygame.font.SysFont('helveticaneue,helvetica,arial', 16)
        self.f_petit = pygame.font.SysFont('helveticaneue,helvetica,arial', 13)
        self.f_mono = pygame.font.SysFont('menlo,dejavusansmono,couriernew', 14)
        self.f_case = pygame.font.SysFont('helveticaneue,helvetica,arial', 11, bold=True)

        self.boutons = []                # rempli au dessin, lu au clic
        self.analyser()

    # --------------- calcul en arriere-plan ---------------

    def lancer(self, cible):
        """Demande un calcul. Un seul tourne a la fois, la fenetre reste vivante."""
        self.attente = cible        # une demande plus recente remplace la precedente
        self.pomper()

    def pomper(self):
        """Demarre la tache en attente des que le thread precedent a fini."""
        if self.attente is None:
            return
        if self.calcul is not None and self.calcul.is_alive():
            return
        cible, self.attente = self.attente, None
        self.calcul = threading.Thread(target=cible, daemon=True)
        self.calcul.start()

    def analyser(self):
        """Evalue la position courante (une passe) pour la barre et le panneau."""
        if self.board.is_game_over(claim_draw=True):
            return

        board = self.board.copy(stack=False)

        def tache():
            _, value, classement = choisir_coup(
                self.model, board, self.device, self.mode, self.args.sims)
            self.value, self.classement = value, classement
            self.classement_fen = board.fen()

        self.lancer(tache)

    def jouer_modele(self):
        board = self.board.copy(stack=False)
        self.reflechit = True

        def tache():
            try:
                coup, value, classement = choisir_coup(
                    self.model, board, self.device,
                    self.mode, self.args.sims, self.args.temp)
                self.value, self.classement = value, classement
                self.classement_fen = board.fen()
                if coup is not None and board.fen() == self.board.fen():
                    self.a_jouer = coup
            finally:
                self.reflechit = False

        self.lancer(tache)

    def demander_indice(self):
        if self.board.turn != self.joueur or self.board.is_game_over(claim_draw=True):
            return
        board = self.board.copy(stack=False)

        def tache():
            coup, value, classement = choisir_coup(
                self.model, board, self.device, self.mode, self.args.sims)
            self.value, self.classement = value, classement
            self.classement_fen = board.fen()
            if coup is not None and board.fen() == self.board.fen():
                self.indice = coup
                self.message = f'indice : {board.san(coup)}'

        self.lancer(tache)

    # --------------- partie ---------------

    def jouer(self, coup):
        self.historique.append(self.board.san(coup))
        self.board.push(coup)
        self.selection = self.indice = None
        self.message = ''

        fin = self.resultat()
        if fin is None:
            self.analyser() if self.board.turn == self.joueur else self.jouer_modele()
        else:
            self.message = fin

    def annuler(self):
        """Annule le coup du modele et le notre, pour revenir a notre trait."""
        if self.reflechit or not self.board.move_stack:
            return
        self.board.pop()                                    # le coup du modele
        self.historique.pop()
        while self.board.move_stack and self.board.turn != self.joueur:
            self.board.pop()                                # puis le notre
            self.historique.pop()
        self.selection = self.indice = None
        self.message = ''
        self.analyser()

    def nouvelle(self):
        if self.reflechit:
            return
        self.board = chess.Board(self.depart_fen)
        self.historique = []
        self.selection = self.indice = self.promotion = None
        self.classement, self.message = [], ''
        if self.board.turn == self.joueur:
            self.analyser()
        else:
            self.jouer_modele()

    def resultat(self):
        outcome = self.board.outcome(claim_draw=True)
        if outcome is None:
            return None
        if outcome.winner is None:
            return f'nulle ({outcome.termination.name.lower()})'
        gagnant = 'blancs' if outcome.winner == chess.WHITE else 'noirs'
        perdant = self.joueur == (not outcome.winner)
        return f'les {gagnant} gagnent — {"tu as perdu" if perdant else "tu as gagne"}'

    def coup_vers(self, depart, arrivee):
        """Le coup legal depart -> arrivee, ou None. Ouvre le choix de promotion."""
        coup = chess.Move(depart, arrivee)
        if coup in self.board.legal_moves:
            return coup

        if chess.Move(depart, arrivee, promotion=chess.QUEEN) in self.board.legal_moves:
            self.promotion = (depart, arrivee)
        return None

    # --------------- geometrie ---------------

    def rect_case(self, case):
        col, rang = chess.square_file(case), chess.square_rank(case)
        if self.bas == chess.WHITE:
            x, y = col, 7 - rang
        else:
            x, y = 7 - col, rang
        return pygame.Rect(self.x0 + x * CASE, self.y0 + y * CASE, CASE, CASE)

    def case_grille(self, col, ligne):
        """Case du damier a la colonne / ligne de l'ecran (0 = en haut a gauche)."""
        if self.bas == chess.WHITE:
            return chess.square(col, 7 - ligne)
        return chess.square(7 - col, ligne)

    def case_sous(self, pos):
        x, y = pos[0] - self.x0, pos[1] - self.y0
        if not (0 <= x < PLATEAU and 0 <= y < PLATEAU):
            return None
        return self.case_grille(int(x // CASE), int(y // CASE))

    @property
    def x0(self):
        return MARGE + BARRE_L + 12

    @property
    def y0(self):
        return MARGE

    def rects_promotion(self):
        """Les quatre choix, empiles depuis la case d'arrivee vers l'interieur."""
        depart, arrivee = self.promotion
        rect = self.rect_case(arrivee)
        vers_bas = rect.y < self.y0 + PLATEAU / 2
        return [pygame.Rect(rect.x, rect.y + (i if vers_bas else -i) * CASE, CASE, CASE)
                for i in range(4)]

    # --------------- entrees ---------------

    def clic(self, pos, presse):
        if self.promotion is not None:
            if presse:
                for piece, rect in zip(PROMOTIONS, self.rects_promotion()):
                    if rect.collidepoint(pos):
                        depart, arrivee = self.promotion
                        self.promotion = None
                        self.jouer(chess.Move(depart, arrivee, promotion=piece))
                        return
                self.promotion = None
            return

        if presse:
            for rect, action in self.boutons:
                if rect.collidepoint(pos):
                    action()
                    return

        if self.reflechit or self.board.turn != self.joueur:
            return
        if self.board.is_game_over(claim_draw=True):
            return

        case = self.case_sous(pos)
        if case is None:
            if presse:
                self.selection = None
            return

        piece = self.board.piece_at(case)

        if presse:
            if self.selection is not None and case != self.selection:
                coup = self.coup_vers(self.selection, case)
                if coup is not None:
                    self.jouer(coup)
                    return
            if piece is not None and piece.color == self.joueur:
                self.selection = case
                self.glisse = case
            else:
                self.selection = None
            return

        # relachement : fin d'un glisser-deposer
        if self.glisse is not None and case != self.glisse:
            coup = self.coup_vers(self.glisse, case)
            self.glisse = None
            if coup is not None:
                self.jouer(coup)
                return
        self.glisse = None

    def touche(self, code):
        if code in (pygame.K_ESCAPE, pygame.K_q):
            return False
        if code == pygame.K_u:
            self.annuler()
        elif code == pygame.K_n:
            self.nouvelle()
        elif code == pygame.K_f:
            self.bas = not self.bas
        elif code == pygame.K_h:
            self.demander_indice()
        elif code == pygame.K_m:
            modes = ['recherche', 'policy', 'value']
            self.mode = modes[(modes.index(self.mode) + 1) % len(modes)]
            self.classement = []
            self.analyser()
        return True

    # --------------- dessin ---------------

    def texte(self, surface, texte, police, couleur, x, y):
        surface.blit(police.render(texte, True, couleur), (x, y))

    def dessiner_piece(self, piece, rect, alpha=255):
        glyphe = GLYPHE[piece.piece_type]
        centre = rect.center

        if piece.color == chess.WHITE:
            corps, contour = (250, 250, 248), (40, 38, 36)
        else:
            corps, contour = (38, 36, 34), (225, 222, 218)

        for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2), (-2, -2), (2, -2), (-2, 2), (2, 2)):
            ombre = self.f_piece.render(glyphe, True, contour)
            ombre.set_alpha(alpha)
            self.ecran.blit(ombre, ombre.get_rect(center=(centre[0] + dx, centre[1] + dy)))

        rendu = self.f_piece.render(glyphe, True, corps)
        rendu.set_alpha(alpha)
        self.ecran.blit(rendu, rendu.get_rect(center=centre))

    def dessiner_plateau(self):
        dernier = self.board.move_stack[-1] if self.board.move_stack else None
        source = self.glisse if self.glisse is not None else self.selection

        for case in chess.SQUARES:
            rect = self.rect_case(case)
            claire = (chess.square_file(case) + chess.square_rank(case)) % 2 == 1
            pygame.draw.rect(self.ecran, CLAIR if claire else FONCE, rect)

            if dernier is not None and case in (dernier.from_square, dernier.to_square):
                voile = pygame.Surface((CASE, CASE), pygame.SRCALPHA)
                voile.fill((*DERNIER, 70))
                self.ecran.blit(voile, rect)

            if case == source:
                voile = pygame.Surface((CASE, CASE), pygame.SRCALPHA)
                voile.fill((*SELECTION, 110))
                self.ecran.blit(voile, rect)

            piece = self.board.piece_at(case)
            if piece is not None and piece.piece_type == chess.KING \
                    and self.board.is_check() and piece.color == self.board.turn:
                pygame.draw.rect(self.ecran, ECHEC, rect, 4)

        # reperes a..h et 1..8, dans les coins des cases de bord et a contre-couleur
        def encre(case):
            claire = (chess.square_file(case) + chess.square_rank(case)) % 2 == 1
            return FONCE if claire else CLAIR

        for i in range(8):
            bas = self.case_grille(i, 7)            # derniere ligne : lettres
            gauche = self.case_grille(0, i)         # premiere colonne : chiffres
            self.texte(self.ecran, chess.FILE_NAMES[chess.square_file(bas)], self.f_case,
                       encre(bas), self.x0 + (i + 1) * CASE - 12, self.y0 + PLATEAU - 16)
            self.texte(self.ecran, chess.RANK_NAMES[chess.square_rank(gauche)], self.f_case,
                       encre(gauche), self.x0 + 5, self.y0 + i * CASE + 5)

        # coups possibles depuis la case selectionnee
        if source is not None:
            for coup in self.board.legal_moves:
                if coup.from_square != source:
                    continue
                rect = self.rect_case(coup.to_square)
                voile = pygame.Surface((CASE, CASE), pygame.SRCALPHA)
                if self.board.piece_at(coup.to_square) is not None:
                    pygame.draw.circle(voile, (20, 20, 20, 90),
                                       (CASE // 2, CASE // 2), CASE // 2 - 3, 6)
                else:
                    pygame.draw.circle(voile, (20, 20, 20, 80),
                                       (CASE // 2, CASE // 2), CASE // 7)
                self.ecran.blit(voile, rect)

        if self.indice is not None:
            for case in (self.indice.from_square, self.indice.to_square):
                pygame.draw.rect(self.ecran, ACCENT, self.rect_case(case), 4)

        for case in chess.SQUARES:
            piece = self.board.piece_at(case)
            if piece is None:
                continue
            if case == self.glisse:
                continue                     # dessinee sous la souris
            self.dessiner_piece(piece, self.rect_case(case))

        if self.glisse is not None:
            piece = self.board.piece_at(self.glisse)
            if piece is not None:
                rect = pygame.Rect(0, 0, CASE, CASE)
                rect.center = self.souris
                self.dessiner_piece(piece, rect)

        pygame.draw.rect(self.ecran, (24, 22, 20),
                         (self.x0 - 2, self.y0 - 2, PLATEAU + 4, PLATEAU + 4), 2)

    def dessiner_promotion(self):
        voile = pygame.Surface((PLATEAU, PLATEAU), pygame.SRCALPHA)
        voile.fill((10, 10, 10, 150))
        self.ecran.blit(voile, (self.x0, self.y0))

        for piece, rect in zip(PROMOTIONS, self.rects_promotion()):
            pygame.draw.rect(self.ecran, (240, 238, 234), rect)
            pygame.draw.rect(self.ecran, (60, 58, 54), rect, 2)
            self.dessiner_piece(chess.Piece(piece, self.joueur), rect)

    def dessiner_barre(self):
        """Barre d'evaluation, blanc en bas si les blancs sont en bas."""
        cible = self.value if self.board.turn == chess.WHITE else -self.value
        self.value_affichee += (cible - self.value_affichee) * 0.15   # lissage

        rect = pygame.Rect(MARGE, self.y0, BARRE_L, PLATEAU)
        pygame.draw.rect(self.ecran, (30, 28, 26), rect)

        part = (self.value_affichee + 1) / 2                # 0 noirs .. 1 blancs
        h = int(PLATEAU * part)
        if self.bas == chess.WHITE:
            blanc = pygame.Rect(rect.x, rect.bottom - h, BARRE_L, h)
        else:
            blanc = pygame.Rect(rect.x, rect.y, BARRE_L, h)
        pygame.draw.rect(self.ecran, (238, 236, 232), blanc)

        pygame.draw.line(self.ecran, (120, 116, 110),
                         (rect.x, rect.centery), (rect.right, rect.centery))
        pygame.draw.rect(self.ecran, (24, 22, 20), rect, 1)

    def bouton(self, rect, libelle, action):
        survol = rect.collidepoint(self.souris)
        pygame.draw.rect(self.ecran, BOUTON_SURVOL if survol else BOUTON, rect, border_radius=6)
        rendu = self.f_petit.render(libelle, True, TEXTE)
        self.ecran.blit(rendu, rendu.get_rect(center=rect.center))
        self.boutons.append((rect, action))

    def dessiner_panneau(self):
        x = self.x0 + PLATEAU + 16
        y = self.y0
        self.boutons = []

        self.texte(self.ecran, 'ChessAI', self.f_titre, TEXTE, x, y)
        y += 34

        cote = 'blancs' if self.joueur == chess.WHITE else 'noirs'
        self.texte(self.ecran, f'tu joues les {cote}  ·  mode {self.mode}  ·  {self.device.type}',
                   self.f_petit, TEXTE_GRIS, x, y)
        y += 26

        # evaluation
        cp = en_centipions(self.value if self.board.turn == chess.WHITE else -self.value)
        self.texte(self.ecran, f'{cp / 100:+.2f}', self.f_titre, TEXTE, x, y)
        self.texte(self.ecran, 'evaluation (point de vue des blancs)',
                   self.f_petit, TEXTE_GRIS, x + 92, y + 10)
        y += 40

        if self.reflechit:
            etat = 'le modele reflechit...'
        elif self.message:
            etat = self.message
        elif self.board.turn == self.joueur:
            etat = 'echec !' if self.board.is_check() else 'a toi de jouer'
        else:
            etat = ''
        self.texte(self.ecran, etat, self.f_texte, ACCENT, x, y)
        y += 30

        # coups envisages
        self.texte(self.ecran, 'coups envisages', self.f_petit, TEXTE_GRIS, x, y)
        y += 20
        a_jour = self.classement_fen == self.board.fen()
        for coup, proba, score in (self.classement[:5] if a_jour else []):
            san = self.board.san(coup)
            note = '' if score is None else f'{en_centipions(score) / 100:+.2f}'
            self.texte(self.ecran, san, self.f_mono, TEXTE, x, y)
            self.texte(self.ecran, f'{proba:5.1%}', self.f_mono, TEXTE_GRIS, x + 72, y)
            self.texte(self.ecran, note, self.f_mono, TEXTE_GRIS, x + 132, y)

            largeur = int(90 * proba)
            pygame.draw.rect(self.ecran, ACCENT, (x + 190, y + 5, max(largeur, 1), 8))
            y += 20
        y += 14

        # historique, deux colonnes de coups
        self.texte(self.ecran, 'partie', self.f_petit, TEXTE_GRIS, x, y)
        y += 20
        haut = y
        lignes = [(i // 2 + 1, self.historique[i], self.historique[i + 1]
                   if i + 1 < len(self.historique) else '')
                  for i in range(0, len(self.historique), 2)]
        n_max = (self.y0 + PLATEAU - 60 - haut) // 18
        for numero, blanc, noir in lignes[-n_max:]:
            self.texte(self.ecran, f'{numero:>3}. {blanc:<8}{noir}', self.f_mono, TEXTE, x, y)
            y += 18

        # boutons
        y = self.y0 + PLATEAU - 34
        largeur = (PANNEAU_L - 3 * 8) // 4
        for i, (libelle, action) in enumerate([
                ('nouvelle', self.nouvelle), ('annuler', self.annuler),
                ('retourner', lambda: setattr(self, 'bas', not self.bas)),
                ('indice', self.demander_indice)]):
            self.bouton(pygame.Rect(x + i * (largeur + 8), y, largeur, 30), libelle, action)

    def dessiner(self):
        self.ecran.fill(FOND)
        self.dessiner_barre()
        self.dessiner_plateau()
        self.dessiner_panneau()
        if self.promotion is not None:
            self.dessiner_promotion()
        pygame.display.flip()

    # --------------- boucle ---------------

    def boucle(self):
        if self.board.turn != self.joueur:
            self.jouer_modele()

        tourne = True
        while tourne:
            for evenement in pygame.event.get():
                if evenement.type == pygame.QUIT:
                    tourne = False
                elif evenement.type == pygame.KEYDOWN:
                    tourne = self.touche(evenement.key)
                elif evenement.type == pygame.MOUSEMOTION:
                    self.souris = evenement.pos
                elif evenement.type == pygame.MOUSEBUTTONDOWN and evenement.button == 1:
                    self.souris = evenement.pos
                    self.clic(evenement.pos, True)
                elif evenement.type == pygame.MOUSEBUTTONUP and evenement.button == 1:
                    self.souris = evenement.pos
                    self.clic(evenement.pos, False)

            self.pomper()

            # le coup du modele est applique ici, dans le thread de la fenetre
            if self.a_jouer is not None:
                coup, self.a_jouer = self.a_jouer, None
                self.jouer(coup)

            self.dessiner()
            self.horloge.tick(60)

        pygame.quit()


def main():
    p = argparse.ArgumentParser(description='Jouer contre le modele (pygame).')
    p.add_argument('--model', default=MODEL_PATH, help='checkpoint .pt')
    p.add_argument('--couleur', default='blanc', choices=['blanc', 'noir', 'hasard'])
    p.add_argument('--mode', default='recherche',
                   choices=['recherche', 'policy', 'value'])
    # REMPLACEMENT DE recherche_2 PAR UN MCTS : --topk (nombre de coups explores
    # a profondeur fixe) n'a plus de sens, PUCT repartit lui-meme le budget.
    p.add_argument('--sims', type=int, default=N_SIMULATIONS,
                   help='simulations MCTS par coup (mode recherche)')
    p.add_argument('--temp', type=float, default=0.0,
                   help='> 0 : echantillonne au lieu de prendre le meilleur coup')
    p.add_argument('--fen', default=None, help='position de depart')
    p.add_argument('--device', default=None, help='cpu / mps / cuda')
    Jeu(p.parse_args()).boucle()


if __name__ == '__main__':
    main()
