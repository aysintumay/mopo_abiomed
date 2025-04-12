import sys
import d4rl
import numpy as np
import csv
import gym

def compute_normalized_score(csv_path, env_name):
    print("Script has started")
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        returns = [float(row['mean_return']) for row in reader]

    normalized_mean_scores = [d4rl.get_normalized_score(env_name, np.array(r)) * 100 for r in returns]
    mean_score = np.mean(normalized_mean_scores)
    std_score = np.std(normalized_mean_scores)

    print(f"Computed Mean Normalized Score: {mean_score:.2f} ± {std_score:.2f}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python normalizer.py <csv_path> <env_name>")
    else:
        csv_path = sys.argv[1]
        env_name = sys.argv[2]
        compute_normalized_score(csv_path, env_name)