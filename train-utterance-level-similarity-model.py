import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import IPython

from dataset.dataset import *
from model.customized_similarity_model import UtteranceLevelModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    random.seed()
    TOP_K = args.top_k

    assert (
        args.utterance_list is not None
    ), "Require csv file of utterance-level similarity. Please run predefined utterance-level MIA first."

    df = pd.read_csv(args.utterance_list, index_col=False)

    # Select the top k utterances from the csv file
    utterances = [x for x in df["Unseen_utterance"].values if str(x) != "nan"]
    similarity = [x for x in df["Unseen_utterance_sim"].values if str(x) != "nan"]
    sorted_similarity, sorted_utterances = zip(*sorted(zip(similarity, utterances)))
    sorted_similarity = list(sorted_similarity)
    sorted_utterances = list(sorted_utterances)

    negative_utterances = sorted_utterances[:TOP_K]
    positive_utterances = sorted_utterances[-TOP_K:]
    train_dataset = CertainUtteranceDataset(
        args.base_path, positive_utterances, negative_utterances, args.model
    )

    eval_negative_utterances = sorted_utterances[TOP_K : 2 * TOP_K]
    eval_positive_utterances = sorted_utterances[-2 * TOP_K : -TOP_K]
    eval_dataset = CertainUtteranceDataset(
        args.base_path, eval_positive_utterances, eval_negative_utterances, args.model,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=train_dataset.collate_fn,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=eval_dataset.collate_fn,
    )

    # Build similarity model
    feature, _ = train_dataset[0]
    input_dim = feature.shape[-1]

    model = UtteranceLevelModel(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    min_loss = 1000
    early_stopping = 0
    epoch = 0
    while epoch < args.n_epochs:

        # Train the model
        model.train()
        for batch_id, (features, labels) in enumerate(
            tqdm(train_dataloader, dynamic_ncols=True, desc=f"Train | Epoch {epoch+1}")
        ):
            features = [torch.FloatTensor(feature).to(device) for feature in features]
            labels = torch.FloatTensor([label for label in labels]).to(device)
            pred = model(features)
            loss = torch.mean(criterion(pred, labels))
            loss.backward()
            optimizer.step()

        # Eval the model
        model.eval()
        total_loss = []
        for batch_id, (features, labels) in enumerate(
            tqdm(eval_dataloader, dynamic_ncols=True, desc="Eval")
        ):
            features = [torch.FloatTensor(feature).to(device) for feature in features]
            labels = torch.FloatTensor([label for label in labels]).to(device)
            with torch.no_grad():
                pred = model(features)

            loss = criterion(pred, labels)
            total_loss += loss.detach().cpu().tolist()

        total_loss = np.mean(total_loss)

        # Check whether to save the model
        if total_loss < min_loss:
            min_loss = total_loss
            print(f"Saving model (epoch = {(epoch + 1):4d}, loss = {min_loss:.4f})")
            torch.save(
                model.state_dict(),
                os.path.join(
                    args.output_path,
                    f"customized-utterance-similarity-model-{args.model}.pt",
                ),
            )
            early_stopping = 0
        else:
            print(
                f"Not saving model (epoch = {(epoch + 1):4d}, loss = {total_loss:.4f})"
            )
            early_stopping = early_stopping + 1

        # Check whether early stopping the training
        if early_stopping < 5:
            epoch = epoch + 1
        else:
            epoch = args.n_epochs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_path", help="directory of feature of LibriSpeech dataset"
    )
    parser.add_argument("--output_path", help="directory to save the model")
    parser.add_argument(
        "--model", help="which self-supervised model you used to extract features"
    )
    parser.add_argument("--seed", type=int, default=57, help="random seed")
    parser.add_argument(
        "--top_k", type=int, default=500, help="how many utterance to pick"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=32, help="training batch size"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=32, help="evaluation batch size"
    )
    parser.add_argument(
        "--utterance_list", type=str, default=None, help="certain utterance list"
    )
    parser.add_argument("--n_epochs", type=int, default=30, help="training epoch")
    parser.add_argument("--num_workers", type=int, default=2, help="number of workers")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    args = parser.parse_args()

    main(args)
