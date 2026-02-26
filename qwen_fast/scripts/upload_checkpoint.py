import argparse
import shutil
import tempfile
from pathlib import Path
from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description="Upload a ContextVLA checkpoint to HuggingFace Hub.")
    parser.add_argument("--repo-id", required=True, help="HuggingFace repo ID, e.g. jasonkongie/contextvla-LIBERO-ctx8")
    parser.add_argument("--checkpoint-dir", required=True, type=Path, help="Path to local checkpoint directory")
    parser.add_argument("--step", required=True, type=int, help="Training step number (used as subfolder name)")
    parser.add_argument("--private", action="store_true", default=False, help="Make the repo private")
    args = parser.parse_args()

    api = HfApi()

    # Create repo if it doesn't already exist
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    print(f"Repo ready: https://huggingface.co/{args.repo_id}")

    # Stage files in a temp dir, skipping optimizer.pt (~24GB, not needed for inference/eval)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / f"step-{args.step}"
        tmp_path.mkdir()

        for item in args.checkpoint_dir.iterdir():
            if item.name == "optimizer.pt":
                print(f"Skipping {item.name} (optimizer state, not needed for eval)")
                continue
            dest = tmp_path / item.name
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)

        print(f"Uploading step-{args.step} to {args.repo_id}/step-{args.step} ...")
        api.upload_folder(
            folder_path=str(tmp_path),
            repo_id=args.repo_id,
            repo_type="model",
            path_in_repo=f"step-{args.step}",
            commit_message=f"Add checkpoint at step {args.step}",
        )

    print(f"Done! https://huggingface.co/{args.repo_id}/tree/main/step-{args.step}")


if __name__ == "__main__":
    main()
