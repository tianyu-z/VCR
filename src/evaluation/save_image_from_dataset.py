from datasets import load_dataset

# If the dataset is gated/private, make sure you have run huggingface-cli login
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save image to dataset.")
    parser.add_argument(
        "--output_path",
        type=str,
        help="folder path",
        default="test_500",
    )
    args = parser.parse_args()
    for lang in ["en", "zh"]:
        for diff in ["easy", "hard"]:
            if not os.path.exists(f"{args.output_path}/{lang}_{diff}"):
                os.makedirs(f"{args.output_path}/{lang}_{diff}")
            dataset = load_dataset(f"vcr-org/VCR-wiki-{lang}-{diff}-test-500")["test"]

            for i in range(500):
                # save image
                dataset[i]["stacked_image"].save(
                    os.path.join(
                        f"{args.output_path}/{lang}_{diff}/stacked_image_{i}.png"
                    )
                )
                dataset[i]["only_it_image"].save(
                    os.path.join(
                        f"{args.output_path}/{lang}_{diff}/only_it_image_{i}.png"
                    )
                )
                dataset[i]["only_it_image_small"].save(
                    os.path.join(
                        f"{args.output_path}/{lang}_{diff}/only_it_image_small_{i}.png"
                    )
                )
                # save text
                with open(
                    os.path.join(f"{args.output_path}/{lang}_{diff}/text_{i}.txt"),
                    "w",
                ) as f:
                    f.write(str(dataset[i]["crossed_text"]))
