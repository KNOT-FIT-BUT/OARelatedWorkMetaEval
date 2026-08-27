# -*- coding: UTF-8 -*-
"""
Created on 25.05.26

Uploads the OARelatedWorkMetaEval dataset to the Hugging Face Hub.

The local ``hf_dataset`` directory holds two subsets saved with
``Dataset.save_to_disk`` (under ``<subset>/train``). This script pushes each of
them as a separate configuration of a single Hub dataset and then replaces the
auto-generated dataset card body with the human-written ``hf_dataset/README.md``
(keeping the ``configs``/``dataset_info`` metadata that ``push_to_hub``
generated).

Authentication uses the standard Hugging Face credentials: either run
``huggingface-cli login`` beforehand, set the ``HF_TOKEN`` environment variable,
or pass ``--token``.

:author:     Martin Dočekal
"""
import argparse
from pathlib import Path

from datasets import load_from_disk
from huggingface_hub import DatasetCard

DEFAULT_REPO_ID = "BUT-FIT/OARelatedWorkMetaEval"
# directory name -> Hub configuration (subset) name
CONFIGS = ["duels", "statements"]
# metadata keys that push_to_hub maintains and that we must not overwrite
GENERATED_METADATA_KEYS = {"dataset_info", "configs"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo_id", default=DEFAULT_REPO_ID,
                        help=f"Target Hub dataset repository (default: {DEFAULT_REPO_ID}).")
    parser.add_argument("--dataset_dir", default="hf_dataset", type=Path,
                        help="Directory containing the <subset>/train folders (default: hf_dataset).")
    parser.add_argument("--readme", default=None, type=Path,
                        help="Dataset card to publish (default: <dataset_dir>/README.md).")
    parser.add_argument("--private", action="store_true",
                        help="Create/keep the repository private.")
    parser.add_argument("--token", default=None,
                        help="Hugging Face token (defaults to the cached login or HF_TOKEN).")
    parser.add_argument("--commit_message", default="Upload OARelatedWorkMetaEval dataset",
                        help="Commit message used for the data pushes.")
    return parser.parse_args()


def push_subsets(repo_id: str, dataset_dir: Path, private: bool, token: str | None,
                 commit_message: str) -> None:
    """Push every subset in ``CONFIGS`` as its own configuration."""
    for config_name in CONFIGS:
        subset_path = dataset_dir / config_name / "train"
        if not subset_path.exists():
            raise FileNotFoundError(f"Expected subset folder not found: {subset_path}")

        print(f"Loading subset '{config_name}' from {subset_path} ...")
        dataset = load_from_disk(str(subset_path))

        print(f"Pushing subset '{config_name}' ({len(dataset)} rows) to {repo_id} ...")
        dataset.push_to_hub(
            repo_id,
            config_name=config_name,
            split="train",
            private=private,
            token=token,
            commit_message=f"{commit_message} ({config_name})",
        )


def push_readme(repo_id: str, readme_path: Path, token: str | None) -> None:
    """Replace the card body with ``readme_path`` while preserving generated metadata."""
    if not readme_path.exists():
        print(f"No README found at {readme_path}; skipping dataset card update.")
        return

    print(f"Updating dataset card from {readme_path} ...")
    # card currently on the Hub (carries dataset_info + configs created by push_to_hub)
    hub_card = DatasetCard.load(repo_id, repo_type="dataset", token=token)
    local_card = DatasetCard.load(readme_path)

    # overlay our descriptive metadata, but keep the generated config metadata
    for key, value in local_card.data.to_dict().items():
        if key not in GENERATED_METADATA_KEYS:
            setattr(hub_card.data, key, value)

    hub_card.text = local_card.text
    hub_card.push_to_hub(repo_id, repo_type="dataset", token=token)


def main() -> None:
    args = parse_args()
    readme_path = args.readme or (args.dataset_dir / "README.md")

    push_subsets(args.repo_id, args.dataset_dir, args.private, args.token, args.commit_message)
    push_readme(args.repo_id, readme_path, args.token)

    print(f"Done. See https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()