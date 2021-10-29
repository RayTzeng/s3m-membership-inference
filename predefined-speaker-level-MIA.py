import argparse
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataset import *
from utils.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    random.seed(args.seed)

    seen_splits = ["train-clean-100"]
    unseen_splits = ["test-clean", "test-other", "dev-clean", "dev-other"]

    # Load the dataset
    seen_dataset = PredefinedSpeakerLevelDataset(
        args.seen_base_path, seen_splits, args.model
    )
    unseen_dataset = PredefinedSpeakerLevelDataset(
        args.unseen_base_path, unseen_splits, args.model
    )

    seen_dataloader = DataLoader(
        seen_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=seen_dataset.collate_fn,
    )
    unseen_dataloader = DataLoader(
        unseen_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=unseen_dataset.collate_fn,
    )

    seen_speaker_sim = defaultdict(float)

    # Calculate similarity scores of seen data
    for batch_id, (speaker_features, speakers) in enumerate(
        tqdm(seen_dataloader, dynamic_ncols=True, desc="Seen")
    ):
        for speaker_feature, speaker in zip(speaker_features, speakers):
            sim = cosine_similarity(speaker_feature)
            sim = sim[np.triu_indices(len(sim), k=1)]
            seen_speaker_sim[speaker] = np.mean(sim)

    unseen_speaker_sim = defaultdict(float)

    # Calculate similarity scores of unseen data
    for batch_id, (speaker_features, speakers) in enumerate(
        tqdm(unseen_dataloader, dynamic_ncols=True, desc="Unseen")
    ):
        for speaker_feature, speaker in zip(speaker_features, speakers):
            sim = cosine_similarity(speaker_feature)
            sim = sim[np.triu_indices(len(sim), k=1)]
            unseen_speaker_sim[speaker] = np.mean(sim)

    # Apply attack according to the similarity scores
    percentile_choice = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    AA, THR = compute_adversarial_advantage_by_percentile(
        list(seen_speaker_sim.values()),
        list(unseen_speaker_sim.values()),
        percentile_choice,
        args.model,
    )

    TPRs, FPRs, avg_AUC, avg, best = compute_adversarial_advantage_by_ROC(
        list(seen_speaker_sim.values()), list(unseen_speaker_sim.values()), args.model
    )

    percentile_choice += ["average", "best"]
    AA += [avg[0], best[0]]
    THR += [avg[1], best[1]]

    # Results
    result_df = pd.DataFrame(
        {"Percentile": percentile_choice, "Adversarial Advantage": AA, "Threshold": THR}
    )

    result_df.to_csv(
        os.path.join(
            args.output_path,
            f"{args.model}-predefined-speaker-level-attack-result.csv",
        ),
        index=False,
    )

    seen_df = pd.DataFrame(
        {
            "Seen_speaker": list(seen_speaker_sim),
            "Seen_speaker_sim": list(seen_speaker_sim.values()),
        }
    )
    unseen_df = pd.DataFrame(
        {
            "Unseen_speaker": list(unseen_speaker_sim),
            "Unseen_speaker_sim": list(unseen_speaker_sim.values()),
        }
    )

    sim_df = pd.concat([seen_df, unseen_df], axis=1)

    sim_df.to_csv(
        os.path.join(
            args.output_path,
            f"{args.model}-predefined-speaker-level-attack-similarity.csv",
        ),
        index=False,
    )

    plt.figure()
    plt.rcParams.update({"font.size": 12})
    plt.title(f"Speaker-level attack ROC Curve - {args.model}")
    plt.plot(
        FPRs, TPRs, color="darkorange", lw=2, label=f"ROC curve (area = {avg_AUC:0.2f})"
    )
    plt.plot([0, 1], [0, 1], color="grey", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend(loc="lower right")
    plt.savefig(
        os.path.join(
            args.output_path,
            f"{args.model}-predefined-speaker-level-attack-ROC-curve.png",
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seen_base_path",
        help="directory of feature of the seen dataset (default LibriSpeech-100)",
    )
    parser.add_argument(
        "--unseen_base_path",
        help="directory of feature of the unseen dataset (default LibriSpeech-[dev/test])",
    )
    parser.add_argument("--output_path", help="directory to save the analysis results")
    parser.add_argument(
        "--model", help="which self-supervised model you used to extract features"
    )
    parser.add_argument("--seed", type=int, default=57, help="random seed")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers")
    args = parser.parse_args()

    main(args)

