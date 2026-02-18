import numpy as np
import itertools
from sklearn.preprocessing import normalize
from gensim.models import KeyedVectors
from multiprocessing import Pool, freeze_support
from tqdm import tqdm
import json

# global variable for worker processes
model_shared = None

def init_worker(model_path):
    # loads model once per process
    global model_shared
    model_shared = KeyedVectors.load_word2vec_format(model_path, binary=True, limit=500000)

def get_word_vectors(words):
    # uses global model
    model = model_shared
    word_vectors = []
    
    sample_key = model.index_to_key[0]
    has_tags = "_" in sample_key

    for word in words:
        clean_word = word.lower()
        found_vector = None
        
        if has_tags:
            guesses = [f"{clean_word}_NOUN", f"{clean_word}_VERB", f"{clean_word}_PROPN"]
            for guess in guesses:
                if guess in model:
                    found_vector = model[guess]
                    break
            if found_vector is None:
                for key in model.key_to_index:
                    if key.startswith(clean_word + "_"):
                        found_vector = model[key]
                        break
        else:
            candidates = [word, word.lower(), word.title()]
            for cand in candidates:
                if cand in model:
                    found_vector = model[cand]
                    break
        
        if found_vector is not None:
            word_vectors.append(found_vector)
        else:
            word_vectors.append(np.zeros(model.vector_size))

    return np.array(word_vectors)

def build_similarity_matrix(words, alpha=0.90, beta=0.10):
    """
    Returns W (NxN) weighted similarity graph.
    alpha: embedding cosine weight
    beta : lexical bonus weight (optional, simple)
    """
    X = get_word_vectors(words)
    Xn = normalize(X)  # safe even with zero vectors; sklearn handles rows of zeros

    # cosine sim since normalized
    sim = Xn @ Xn.T
    np.fill_diagonal(sim, 0.0)

    # normalize off-diagonal to [0,1]
    n = sim.shape[0]
    mask = ~np.eye(n, dtype=bool)
    vals = sim[mask]
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo > 1e-12:
        sim_n = (sim - lo) / (hi - lo)
    else:
        sim_n = np.zeros_like(sim)

    # simple lexical bonus (expand later)
    lex = np.zeros((n, n), dtype=np.float32)
    wl = [w.lower() for w in words]
    for i in range(n):
        for j in range(i+1, n):
            b = 0.0
            # same prefix/suffix of len 3
            if len(wl[i]) >= 3 and len(wl[j]) >= 3:
                if wl[i][:3] == wl[j][:3]: b += 1.0
                if wl[i][-3:] == wl[j][-3:]: b += 1.0
            # crude plural
            if (wl[i].endswith("s") and wl[i][:-1] == wl[j]) or (wl[j].endswith("s") and wl[j][:-1] == wl[i]):
                b += 1.0
            lex[i, j] = lex[j, i] = b

    if lex.max() > 0:
        lex = lex / lex.max()

    W = alpha * sim_n + beta * lex
    np.fill_diagonal(W, 0.0)
    return W


def score_group(group, W, outside_lambda=0.35):
    """
    group: tuple of 4 indices.
    cohesion: mean internal edge weight (6 edges)
    outside penalty: average similarity to nodes outside group
    """
    g = list(group)
    pairs = list(itertools.combinations(g, 2))
    cohesion = float(np.mean([W[i, j] for i, j in pairs]))

    n = W.shape[0]
    outside = [k for k in range(n) if k not in g]
    if outside:
        outside_pen = float(np.mean([W[i, outside].mean() for i in g]))
    else:
        outside_pen = 0.0

    return cohesion - outside_lambda * outside_pen

def best_partition_indices(
    W,
    beam_size=250,
    outside_lambda=0.35,
    top_groups_limit=2500,
    banned_group_index_sets=None,   # set[frozenset[int]]
):
    """
    Beam search for 4 disjoint groups maximizing sum of group scores,
    optionally excluding banned 4-sets.
    Returns list of 4 groups (tuples of indices).
    """
    if banned_group_index_sets is None:
        banned_group_index_sets = set()

    n = W.shape[0]
    all_groups = list(itertools.combinations(range(n), 4))

    # pre-score groups (skip banned)
    scored = []
    for g in all_groups:
        if frozenset(g) in banned_group_index_sets:
            continue
        scored.append((g, score_group(g, W, outside_lambda)))

    if not scored:
        return None

    scored.sort(key=lambda x: x[1], reverse=True)
    scored = scored[:min(top_groups_limit, len(scored))]

    def mask_of(g):
        m = 0
        for i in g:
            m |= 1 << i
        return m

    group_masks = {g: mask_of(g) for g, _ in scored}

    beam = [(0, [], 0.0)]
    for step in range(4):
        nxt = []
        for used, chosen, total in beam:
            for g, s in scored:
                m = group_masks[g]
                if used & m:
                    continue
                nxt.append((used | m, chosen + [g], total + s))

        if not nxt:
            return None

        nxt.sort(key=lambda x: x[2], reverse=True)
        beam = nxt[:beam_size]

    best = max(beam, key=lambda x: x[2])
    return best[1]


def solve_board_graph(board_words, banned_groups=None, **kwargs):
    """
    Solve by global partition, excluding any banned groups.

    banned_groups: iterable of sets/lists/tuples of WORD STRINGS (size 4),
                  e.g. [{ "PUPPY","KITTEN","FOAL","CUB" }, ...]
                  If you stored banned as sets of words like your old code,
                  you can pass it directly.

    kwargs are forwarded to best_partition_indices (beam_size, outside_lambda, etc.)
    """
    if banned_groups is None:
        banned_groups = []

    # map word -> indices (handle duplicates safely by storing all positions)
    positions = {}
    for idx, w in enumerate(board_words):
        positions.setdefault(w, []).append(idx)

    banned_index_sets = set()
    for bg in banned_groups:
        bg_list = list(bg)
        if len(bg_list) != 4:
            continue

        # Convert word-set -> a specific index-set.
        # Assumes no duplicates on board (true for Connections).
        idxs = []
        ok = True
        for w in bg_list:
            if w not in positions or not positions[w]:
                ok = False
                break
            idxs.append(positions[w][0])
        if ok:
            banned_index_sets.add(frozenset(idxs))

    W = build_similarity_matrix(board_words)
    groups = best_partition_indices(W, banned_group_index_sets=banned_index_sets, **kwargs)
    if groups is None:
        return None
    return [[board_words[i] for i in g] for g in groups]

def solve_puzzle_worker(p):
    correct_sets = []
    board_words = []

    for ans in p['answers']:
        members = ans['members']
        board_words.extend(members)
        correct_sets.append(set(members))

    # predict full partition
    pred_groups = solve_board_graph(board_words)
    if pred_groups is None:
        return (0, 0, None)

    pred_sets = [set(g) for g in pred_groups]

    categories_found = sum(1 for s in pred_sets if s in correct_sets)
    win = 1 if categories_found == 4 else 0

    perfect_data = p if win == 1 else None
    return (win, categories_found, perfect_data)



def benchmark_multiprocess(json_path, vec_path):
    print(f"Loading puzzles from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        puzzles = json.load(f)
        
    total_puzzles = len(puzzles)
    print(f"Starting Multiprocessing Benchmark on {total_puzzles} puzzles...")
    
    # limits processes to 6 to stay safe
    num_workers = 8
    
    wins = 0
    total_cats = 0
    perfect_puzzles = []
    
    # creates pool
    with Pool(processes=num_workers, initializer=init_worker, initargs=(vec_path,)) as pool:
        # runs puzzles in parallel
        results = list(tqdm(pool.imap(solve_puzzle_worker, puzzles), total=total_puzzles))
        
    # sums results and collects perfects
    for w, c, p_data in results:
        wins += w
        total_cats += c
        if p_data is not None:
            perfect_puzzles.append(p_data)

    # saves perfect puzzles
    with open('perfect_puzzles.json', 'w', encoding='utf-8') as f:
        json.dump(perfect_puzzles, f, indent=4)

    # calculates percentages
    total_possible_cats = total_puzzles * 4
    cat_percentage = (total_cats / total_possible_cats) * 100
    win_percentage = (wins / total_puzzles) * 100

    print("\n" + "="*40)
    print(f"Total Puzzles:      {total_puzzles}")
    print(f"Total Wins:         {wins}")
    print(f"Game Win Rate:      {win_percentage:.2f}%")
    print(f"Categories Found:   {total_cats}/{total_possible_cats}")
    print(f"Category Accuracy:  {cat_percentage:.2f}%")
    print(f"Perfect Games:      {len(perfect_puzzles)}")
    print(f"Saved to:           perfect_puzzles.json")
    print("="*40)


if __name__ == "__main__":
    freeze_support() # needed for windows
    
    vec_path = input("Input model dir: ").strip('"').strip("'")
    benchmark_multiprocess('connections.json', vec_path)