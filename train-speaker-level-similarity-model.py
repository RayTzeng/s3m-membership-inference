import argparse
import os
import random

import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import IPython

from dataset.dataset import *
from model.customized_similarity_model import SpeakerLevelModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    random.seed(args.seed)
    TOP_K = args.top_k

    assert (
        args.speaker_list is not None
    ), "Require csv file of speaker-level similarity. Please run predefined speaker-level MIA first."

    df = pd.read_csv(args.speaker_list, index_col=False)

    # Select the top k speaker from the csv file
    speakers = [x for x in df["Unseen_speaker"].values if str(x) != "nan"]
    similarity = [x for x in df["Unseen_speaker_sim"].values if str(x) != "nan"]
    sorted_similarity, sorted_speakers = zip(*sorted(zip(similarity, speakers)))
    sorted_similarity = list(sorted_similarity)
    sorted_speakers = list(sorted_speakers)
    negative_speakers = sorted_speakers[:TOP_K]
    positive_speakers = sorted_speakers[-TOP_K:]
    train_dataset = CertainSpeakerDataset(
        args.base_path, positive_speakers, negative_speakers, args.model
    )

    eval_negative_speakers = sorted_speakers[TOP_K : 2 * TOP_K]
    eval_positive_speakers = sorted_speakers[-2 * TOP_K : -TOP_K]
    eval_dataset = CertainSpeakerDataset(
        args.base_path, eval_positive_speakers, eval_negative_speakers, args.model
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

    # Build the similarity model
    feature, _, _, _ = train_dataset[0]
    input_dim = feature.shape[-1]
    print(f"input dimension: {input_dim}")

    model = SpeakerLevelModel(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    min_loss = 1000
    early_stopping = 0
    epoch = 0
    while epoch < args.n_epochs:

        # Trian the model
        model.train()
        for batch_id, (features_x, features_y, labels, speakers) in enumerate(
            tqdm(train_dataloader, dynamic_ncols=True, desc=f"Train | Epoch {epoch+1}")
        ):
            optimizer.zero_grad()
            features_x = [
                torch.FloatTensor(feature).to(device) for feature in features_x
            ]
            features_y = [
                torch.FloatTensor(feature).to(device) for feature in features_y
            ]
            labels = torch.FloatTensor([label for label in labels]).to(device)
            pred = model(features_x, features_y)
            loss = torch.mean(criterion(pred, labels))
            loss.backward()
            optimizer.step()

        # Evaluate the model
        model.eval()
        total_loss = []
        for batch_id, (features_x, features_y, labels, speakers) in enumerate(
            tqdm(eval_dataloader, dynamic_ncols=True, desc="Eval")
        ):
            features_x = [
                torch.FloatTensor(feature).to(device) for feature in features_x
            ]
            features_y = [
                torch.FloatTensor(feature).to(device) for feature in features_y
            ]
            labels = torch.FloatTensor([label for label in labels]).to(device)
            with torch.no_grad():
                pred = model(features_x, features_y)

            loss = criterion(pred, labels)
            total_loss += loss.detach().cpu().tolist()

        total_loss = np.mean(total_loss)

        # Check whether to save the model or not
        if total_loss < min_loss:
            min_loss = total_loss
            print(f"Saving model (epoch = {(epoch + 1):4d}, loss = {min_loss:.4f})")
            torch.save(
                model.state_dict(),
                os.path.join(
                    args.output_path,
                    f"customized-speaker-similarity-model-{args.model}.pt",
                ),
            )
            early_stopping = 0

        else:
            print(
                f"Not saving model (epoch = {(epoch + 1):4d}, loss = {total_loss:.4f})"
            )
            early_stopping = early_stopping + 1

        # Check whether early stopping the training or not
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
        "--top_k", type=int, default=1, help="how many speaker to pick",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=32, help="training batch size"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=32, help="evaluation batch size"
    )
    parser.add_argument(
        "--speaker_list", type=str, default=None, help="certain speaker list"
    )
    parser.add_argument("--n_epochs", type=int, default=30, help="training epoch")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--num_workers", type=int, default=2, help="number of workers")
    args = parser.parse_args()

    main(args)
