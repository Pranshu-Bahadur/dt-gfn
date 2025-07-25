import argparse
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from src.wrapper import DTGFNClassifier
import torch
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo # Import the new library

def run_benchmark(dataset_name: str):
    """
    Runs a classification benchmark on a specified dataset over multiple random seeds.
    """
    if dataset_name == "iris":
        X, y = load_iris(return_X_y=True, as_frame=True)
    elif dataset_name == "wine":
        X, y = load_wine(return_X_y=True, as_frame=True)
    elif dataset_name == "breast_cancer":
        X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    elif dataset_name == "raisin":
        # Fetch the Raisin dataset using the robust ucimlrepo library
        try:
            raisin_data = fetch_ucirepo(id=850)
            X = raisin_data.data.features
            y = raisin_data.data.targets
            # The target is categorical, so we encode it to integers
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y.values.ravel()), name='target')
        except Exception as e:
            print(f"Failed to load Raisin dataset using ucimlrepo. Error: {e}")
            print("Please ensure you have run 'pip install ucimlrepo'")
            return
    elif dataset_name == "covertype":
        covtype_data = fetch_covtype(return_X_y=True, as_frame=True)
        X = covtype_data.data
        y = covtype_data.target - 1
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"--- Benchmarking on {dataset_name} ---")
    print(f"Data shape: {X.shape}")
    print(f"Number of classes: {y.nunique()}")

    accuracies = []
    f1_scores = []
    seeds = [1, 2, 3, 4, 5]

    for i, seed in enumerate(seeds):
        print(f"\n--- Running Seed {i+1}/{len(seeds)} (random_state={seed}) ---")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y if y.nunique() > 1 else None)

        model = DTGFNClassifier(
            n_bins=99,
            updates=100,
            rollouts=10,
            batch_size=630,
            top_k_trees=1,
            max_depth=5,
            boosting_lr=1.0,
            reward_function='gini',
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        model.fit(X_train, y_train)
        #model._trainer.cfg.boosting_lr = 0.5
        preds = model.predict(X_test, 'policy', 1000)

        accuracy = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted')

        accuracies.append(accuracy)
        f1_scores.append(f1)

        print(f"Seed {seed} Accuracy: {accuracy:.4f}")
        print(f"Seed {seed} F1 Score (Weighted): {f1:.4f}")

    # --- Final Results ---
    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    avg_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)

    print("\n" + "="*40)
    print("           Final Average Results")
    print("="*40)
    print(f"Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f} ± {std_f1:.4f}")
    print("="*40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="iris",
                        choices=["iris", "wine", "breast_cancer", "raisin", "covertype"])
    args = parser.parse_args()
    run_benchmark(args.dataset)
