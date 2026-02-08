import numpy as np
from gensim.models import KeyedVectors
from k_means_constrained import KMeansConstrained
from sklearn.preprocessing import normalize
import json
import random
from tqdm import tqdm  # Import tqdm for progress bar

def get_predicted_groups(words, model):
    # Loads the vector path
    word_vectors = []
    valid_words = []

    # Checks if word is tagged with noun/adj/etc
    sample_key = model.index_to_key[0]
    has_tags = "_" in sample_key

    # Vectorizes word
    for word in words:
        # Standardize input
        clean_word = word.lower()
        found_vector = None
        
        # Attempts to lookup the word by finding it off the tag
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
            # Standard lookup
            candidates = [word, word.lower(), word.title()]
            for cand in candidates:
                if cand in model:
                    found_vector = model[cand]
                    break
        
        # Error handling if word is not found
        if found_vector is not None:
            word_vectors.append(found_vector)
            valid_words.append(word)
        else:
            # print(f"'{word}' not found. Using Zero Vector.")
            word_vectors.append(np.zeros(model.vector_size))
            valid_words.append(word)

    # Convert list of arrays to a single numpy matrix (N x 300)
    X = np.array(word_vectors)

    # Normalize using cosine similarity
    X_norm = normalize(X)

    best_inertia = float('inf')
    best_labels = None

    for i in range(250):
        # We use a different random_state each time
        clf = KMeansConstrained(
            n_clusters=4,
            size_min=4,
            size_max=4,
            random_state=i 
        )
        clf.fit(X_norm)
        
        # Inertia is how tightly clumped the clusters are
        if clf.inertia_ < best_inertia:
            best_inertia = clf.inertia_
            best_labels = clf.labels_
    
    clusters = {0: [], 1: [], 2: [], 3: []}
    
    for word, label in zip(valid_words, best_labels):
        clusters[label].append(word)

    return list(clusters.values())

def solve_single_game(vec_path):
    model = KeyedVectors.load_word2vec_format(vec_path, binary=True, limit=500000)

    print("Enter 16 words separated by commas.")
    user_input = input("Words: ")
    
    # Process input: split by comma and strip whitespace
    puzzle_words = [w.strip() for w in user_input.split(',')]
    
    groups = get_predicted_groups(puzzle_words, model)

    for i, group in enumerate(groups):
        print(f"Group {i+1}: {group}")

def benchmark_test(json_path, vec_path):
    print(f"Loading puzzles from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        puzzles = json.load(f)
        
    print(f"Loading Word2Vec model from {vec_path}...")
    model = KeyedVectors.load_word2vec_format(vec_path, binary=True)
    
    total_puzzles = len(puzzles)
    total_groups_checked = 0
    total_correct_groups = 0
    perfect_puzzles = 0
    
    print(f"\nStarting benchmark on {total_puzzles} puzzles...")
    
    for p in tqdm(puzzles, desc="Processing Puzzles", unit="game"):
        # 1. Flatten the answers to get the board
        correct_groups_sets = []
        puzzle_words = []
        
        for ans in p['answers']:
            group_members = ans['members']
            puzzle_words.extend(group_members)
            correct_groups_sets.append(set(group_members))
        
        # Shuffle words to simulate real gameplay (prevent order bias)
        random.shuffle(puzzle_words)
        
        # Run Solver
        predicted_groups = get_predicted_groups(puzzle_words, model)
        
        # Score it
        predicted_sets = [set(g) for g in predicted_groups]
        
        matches_this_puzzle = 0
        for p_set in predicted_sets:
            if p_set in correct_groups_sets:
                matches_this_puzzle += 1
        
        total_correct_groups += matches_this_puzzle
        total_groups_checked += 4
        
        if matches_this_puzzle == 4:
            perfect_puzzles += 1
            tqdm.write(f"\nPerfect Game: ")
            for i, grp in enumerate(predicted_groups):
                tqdm.write(f"  Group {i+1}: {grp}")
            tqdm.write("-" * 30)

    accuracy = (total_correct_groups / total_groups_checked) * 100
    perfect_rate = (perfect_puzzles / total_puzzles) * 100
    
    print("\n" + "="*40)
    print("       BENCHMARK RESULTS       ")
    print("="*40)
    print(f"Total Puzzles:      {total_puzzles}")
    print(f"Total Groups:       {total_groups_checked}")
    print(f"Correct Groups:     {total_correct_groups}")
    print(f"Group Accuracy:     {accuracy:.2f}%")
    print(f"Perfect Games:      {perfect_puzzles} ({perfect_rate:.2f}%)")
    print("="*40)

if __name__ == "__main__":
    vec_path = input("Input model dir: ").strip('"').strip("'")
    
    print("\nSelect Mode:")
    print("1. Run Benchmark (puzzles.json)")
    print("2. Solve Single Game (Input words manually)")
    choice = input("Enter 1 or 2: ")
    
    if choice == "1":
        benchmark_test('connections.json', vec_path)
    elif choice == "2":
        solve_single_game(vec_path)
    else:
        print("Invalid choice.")