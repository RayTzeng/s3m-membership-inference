import glob
import math
import os
import random

import numpy as np
import torch
from torchaudio.sox_effects import apply_effects_file
from torch.utils.data.dataset import Dataset
from collections import defaultdict
from tqdm import tqdm
import IPython


class PredefinedSpeakerLevelDataset(Dataset):
    """
        Speaker level dataset for predefined similarity metric function
    """

    def __init__(self, base_path, splits, model):
        self.speakers = self._getspeakerlist(base_path, splits)
        self.model = model

    def __len__(self):
        return len(self.speakers)

    def __getitem__(self, idx):
        speaker_feature = []
        for feature_path in glob.glob(
            os.path.join(self.speakers[idx], "**", f"{self.model}-*"), recursive=True
        ):
            feature = torch.load(feature_path).detach().cpu()
            feature = feature.squeeze()
            speaker_feature.append(np.array(feature).mean(axis=0))
        return speaker_feature, self.speakers[idx]

    def collate_fn(self, samples):
        return zip(*samples)

    def _getspeakerlist(self, base_path, splits):
        speaker_list = []
        split_pathes = [os.path.join(base_path, split) for split in splits]

        for split_path in split_pathes:
            all_speakers = glob.glob(os.path.join(split_path, "*[!.txt]"))
            speaker_list += all_speakers

        return speaker_list


class PredefinedUtteranceLevelDataset(Dataset):
    """
        Utterance level dataset for predefined similarity metric function
    """

    def __init__(self, base_path, splits, model):
        self.utterances = self._getutterancelist(base_path, splits, model)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        feature = torch.load(self.utterances[idx]).detach().cpu()
        feature = feature.squeeze()
        return feature, self.utterances[idx]

    def collate_fn(self, samples):
        return zip(*samples)

    def _getutterancelist(self, base_path, splits, model):
        utterance_list = []
        split_pathes = [os.path.join(base_path, split) for split in splits]

        for split_path in split_pathes:
            split_utterance_list = []
            for speaker in glob.glob(os.path.join(split_path, "*[!.txt]")):
                for feature_path in glob.glob(
                    os.path.join(speaker, "**", f"{model}-*"), recursive=True
                ):
                    split_utterance_list.append(feature_path)
            utterance_list += split_utterance_list

        return utterance_list


class CustomizedSpeakerLevelDataset(Dataset):
    """
        Speaker level dataset for customized similarity metric function
    """

    def __init__(self, base_path, splits, model):
        self.data = self._getdatalist(base_path, splits, model)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature_x_path, feature_y_path, speaker = self.data[idx]
        feature_x = torch.load(feature_x_path).detach().cpu()
        feature_x = feature_x.squeeze()
        feature_y = torch.load(feature_y_path).detach().cpu()
        feature_y = feature_y.squeeze()

        return feature_x.numpy(), feature_y.numpy(), speaker

    def collate_fn(self, samples):

        features_x, features_y, speakers = [], [], []

        for feature_x, feature_y, speaker in samples:
            features_x.append(feature_x)
            features_y.append(feature_y)
            speakers.append(speaker)

        return features_x, features_y, speakers

    def _getdatalist(self, base_path, splits, model):
        data_list = []

        split_pathes = [os.path.join(base_path, split) for split in splits]

        for split_path in split_pathes:
            all_speakers = glob.glob(os.path.join(split_path, "*[!.txt]"))
            for speaker in all_speakers:
                for chapter in glob.glob(os.path.join(speaker, "*")):
                    feature_pathes = glob.glob(os.path.join(chapter, f"{model}-*"))
                    for i in range(len(feature_pathes) - 1):
                        data_list.append(
                            (
                                feature_pathes[i],
                                feature_pathes[i + 1],
                                speaker.split("/")[-1],
                            )
                        )

        return data_list


class CustomizedUtteranceLevelDataset(Dataset):
    """
        Utterance level dataset for customized similarity metric function
    """

    def __init__(self, base_path, splits, model):
        self.utterances = self._getutterancelist(base_path, splits, model)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        feature_path = self.utterances[idx]
        feature = torch.load(feature_path).detach().cpu()
        feature = feature.squeeze()
        return feature, self.utterances[idx]

    def collate_fn(self, samples):
        features, utterances = [], []

        for feature, utterance in samples:
            features.append(feature)
            utterances.append(utterance)

        return features, utterances

    def _getutterancelist(self, base_path, splits, model):
        utterance_list = []

        split_pathes = [os.path.join(base_path, split) for split in splits]

        for split_path in split_pathes:
            split_utterance_list = []
            for speaker in glob.glob(os.path.join(split_path, "*[!.txt]")):
                for chapter in glob.glob(os.path.join(speaker, "*")):
                    for feature_path in glob.glob(os.path.join(chapter, f"{model}-*")):
                        split_utterance_list.append((feature_path))
            utterance_list += split_utterance_list

        # print(len(speaker_list))
        return utterance_list


class CertainSpeakerDataset(Dataset):
    def __init__(self, base_path, seen_speakers, unseen_speakers, model):
        self.data = self._getdatalist(base_path, seen_speakers, unseen_speakers, model)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature_x_path, feature_y_path, label, speaker = self.data[idx]
        feature_x = torch.load(feature_x_path).detach().cpu()
        feature_x = feature_x.squeeze()
        feature_y = torch.load(feature_y_path).detach().cpu()
        feature_y = feature_y.squeeze()

        return feature_x.numpy(), feature_y.numpy(), label, speaker

    def collate_fn(self, samples):
        features_x, features_y, labels, speakers = [], [], [], []

        for feature_x, feature_y, label, speaker in samples:
            features_x.append(feature_x)
            features_y.append(feature_y)
            labels.append(label)
            speakers.append(speaker)

        return features_x, features_y, labels, speakers

    def _getdatalist(self, base_path, seen_speakers, unseen_speakers, model):
        data_list = []

        for speaker in seen_speakers:
            for chapter in glob.glob(os.path.join(speaker, "*")):
                feature_pathes = glob.glob(os.path.join(chapter, f"{model}-*"))
                for i in range(len(feature_pathes) - 1):
                    data_list.append(
                        (
                            feature_pathes[i],
                            feature_pathes[i + 1],
                            1,
                            speaker.split("/")[-1],
                        )
                    )

        for speaker in unseen_speakers:
            for chapter in glob.glob(os.path.join(speaker, "*")):
                feature_pathes = glob.glob(os.path.join(chapter, f"{model}-*"))
                for i in range(len(feature_pathes) - 1):
                    data_list.append(
                        (
                            feature_pathes[i],
                            feature_pathes[i + 1],
                            0,
                            speaker.split("/")[-1],
                        )
                    )

        return data_list


class CertainUtteranceDataset(Dataset):
    def __init__(self, base_path, seen_utterances, unseen_utterances, model):
        self.utterances = self._getutterancelist(
            base_path, seen_utterances, unseen_utterances, model
        )

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        feature_path, label = self.utterances[idx]
        feature = torch.load(feature_path).detach().cpu()
        feature = feature.squeeze()
        # print(len(speaker_feature))
        return feature, label

    def collate_fn(self, samples):
        features, labels = [], []

        for feature, label in samples:
            features.append(feature)
            labels.append(label)

        return features, labels

    def _getutterancelist(self, base_path, seen_utterances, unseen_utterances, model):
        utterance_list = []

        for utterance in seen_utterances:
            utterance_list.append((utterance, 1))

        for utterance in unseen_utterances:
            utterance_list.append((utterance, 0))

        # print(len(speaker_list))
        return utterance_list
