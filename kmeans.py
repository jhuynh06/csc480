import numpy as np
from gensim.models import KeyedVectors
from k_means_constrained import KMeansConstrained
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import euclidean_distances
import json
import random
from tqdm import tqdm
from multiprocessing import Pool, freeze_support

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

def find_best_group(current_words, banned_groups):
    # safety check
    if len(current_words) < 4: return None

    X = get_word_vectors(current_words)
    X_norm = normalize(X)
    
    n_clusters = len(current_words) // 4
    
    best_inertia = float('inf')
    best_group = None

    for i in range(2):
        clf = KMeansConstrained(
            n_clusters=n_clusters,
            size_min=4,
            size_max=4,
            random_state=i 
        )
        clf.fit(X_norm)
        
        centers = clf.cluster_centers_
        labels = clf.labels_
        
        for label_id in range(n_clusters):
            indices = np.where(labels == label_id)[0]
            group_words = [current_words[idx] for idx in indices]
            group_set = set(group_words)
            
            # check banned
            is_banned = False
            for banned in banned_groups:
                if group_set == banned:
                    is_banned = True
                    break
            if is_banned: continue
            
            # calc inertia
            group_vectors = X_norm[indices]
            center = centers[label_id].reshape(1, -1)
            dists = euclidean_distances(group_vectors, center)
            inertia = np.sum(dists ** 2)
            
            if inertia < best_inertia:
                best_inertia = inertia
                best_group = group_words

    return best_group

def solve_puzzle_worker(p):
    # plays one game
    correct_sets = []
    board_words = []
    
    for ans in p['answers']:
        members = ans['members']
        board_words.extend(members)
        correct_sets.append(set(members))
        
    random.shuffle(board_words)
    
    lives = 3
    banned_groups = []
    groups_found = 0
    categories_found = 0
    
    while lives > 0 and groups_found < 4:
        guess = find_best_group(board_words, banned_groups)
        
        if guess is None: break
        
        guess_set = set(guess)
        
        is_correct = False
        for correct in correct_sets:
            if guess_set == correct:
                is_correct = True
                break
        
        if is_correct:
            groups_found += 1
            categories_found += 1
            board_words = [w for w in board_words if w not in guess_set]
        else:
            lives -= 1
            banned_groups.append(guess_set)
            
    win = 1 if groups_found == 4 else 0
    
    # checks for perfect game
    perfect_data = None
    if win == 1 and lives == 3:
        perfect_data = p
        
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