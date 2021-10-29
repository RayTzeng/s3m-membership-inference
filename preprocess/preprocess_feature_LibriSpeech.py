import argparse
import glob
import os

import torch
from torchaudio.sox_effects import apply_effects_file
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    model = torch.hub.load("s3prl/s3prl", args.model).to(device)
    model = model.to(device)
    model.eval()

    if args.output_prefix is not None:
        output_prefix = args.output_prefix
    else:
        output_prefix = args.model

    split_path = os.path.join(args.base_path, args.split)

    speaker_count = 0
    audio_count = 0

    for speaker in tqdm(
        glob.glob(os.path.join(split_path, "*[!.txt]")), ascii=True, desc="Speaker"
    ):
        speaker_count += 1
        for chapter in glob.glob(os.path.join(split_path, speaker, "*")):

            output_folder = os.path.join(
                args.output_path,
                args.split,
                speaker.split("/")[-1],
                chapter.split("/")[-1],
            )

            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
                tqdm.write(f"Directory {output_folder} Created ")

            with torch.no_grad():
                for audio_path in glob.glob(
                    os.path.join(split_path, speaker, chapter, "*.flac")
                ):
                    audio_count += 1

                    audio_name = audio_path.split("/")[-1]

                    wav, _ = apply_effects_file(
                        audio_path, [["channels", "1"], ["rate", "16000"], ["norm"],],
                    )
                    wav = wav.squeeze(0).to(device)

                    feature = model([wav])["last_hidden_state"]

                    output_path = os.path.join(
                        output_folder, f"{output_prefix}-{audio_name}.pt",
                    )
                    tqdm.write(output_path)
                    torch.save(feature.cpu(), output_path)

    print("There are {} speakers".format(speaker_count))
    print("There are {} audios".format(audio_count))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", help="directory of LibriSpeech dataset")
    parser.add_argument("--split", default="wav48")
    parser.add_argument("--output_path", help="directory to save feautures")
    parser.add_argument("--output_prefix", help="prefix of the output filename")
    parser.add_argument(
        "--model", help="which self-supervised model to extract features"
    )
    args = parser.parse_args()

    main(args)
