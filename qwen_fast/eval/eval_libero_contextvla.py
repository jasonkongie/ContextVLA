"""
LIBERO Evaluation Script for ContextVLA
========================================
Runs rollout evaluation of a fine-tuned ContextVLA checkpoint on LIBERO task suites.
No training required — uses the existing ContextVLAPolicy for direct in-process inference.

Adapted from: https://github.com/starVLA/starVLA/tree/starVLA/examples/LIBERO/eval_files

Usage:
    cd qwen_fast/
    python eval/eval_libero_contextvla.py \
        --ckpt-path /path/to/checkpoint \
        --task-suite-name libero_spatial \
        --num-trials-per-task 20 \
        --num-frames 8 \
        --video-out-path eval_videos/ctx8_spatial
"""

import dataclasses
import json
import logging
import math
import os
import pathlib
import sys
from collections import deque

import cv2
import imageio
import numpy as np
import tqdm
import tyro

# ---------------------------------------------------------------------------
# Make sure we can import from the qwen_fast/ src tree regardless of cwd.
# IMPORTANT: append (not insert at 0) so site-packages takes priority,
# preventing qwen_fast/ from shadowing installed packages like libero.
# ---------------------------------------------------------------------------
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from libero.libero import benchmark as _libero_benchmark
from libero.libero import get_libero_path as _get_libero_path
from libero.libero.envs import OffScreenRenderEnv as _OffScreenRenderEnv
from src.policies.contextvla_policy import ContextVLAPolicy

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]   # no-op used during stabilization wait
LIBERO_ENV_RESOLUTION = 256                  # resolution used during data collection
IMAGE_SIZE = 224                             # model input size

MAX_STEPS = {
    "libero_spatial": 220,   # longest training demo has 193 steps
    "libero_object":  280,   # longest training demo has 254 steps
    "libero_goal":    300,   # longest training demo has 270 steps
    "libero_10":      520,   # longest training demo has 505 steps
    "libero_90":      400,   # longest training demo has 373 steps
}


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Args:
    # ---- required ----
    ckpt_path: str
    """Path to the fine-tuned ContextVLA checkpoint directory."""

    # ---- data / model ----
    norm_stats_path: str = "assets/libero/norm_stats.json"
    """Path to norm_stats JSON (relative to qwen_fast/ or absolute)."""

    num_frames: int = 8
    """Context window size — must match the checkpoint that was trained."""

    action_horizon: int = 10
    """Number of actions predicted per inference call."""

    action_dim: int = 7
    """Action dimensionality (7-DOF for Libero)."""

    # ---- eval ----
    task_suite_name: str = "libero_spatial"
    """LIBERO task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90."""

    num_trials_per_task: int = 20
    """Number of rollout episodes per task."""

    num_steps_wait: int = 10
    """Steps to wait at episode start for objects to stabilise."""

    # ---- output ----
    video_out_path: str = "eval_videos"
    """Directory to save per-episode rollout videos."""

    seed: int = 7
    """Random seed for reproducibility."""

    device: str = "cuda"
    """Device to run the model on."""

    save_videos: bool = True
    """Whether to save MP4 rollout videos."""


# ---------------------------------------------------------------------------
# Helpers (ported from starVLA / robosuite)
# ---------------------------------------------------------------------------
def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion to axis-angle (from robosuite transform_utils)."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] ** 2)
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _binarize_gripper(open_val: float) -> float:
    """Map continuous gripper value → {+1.0 open, -1.0 closed}."""
    return 1.0 - 2.0 * (float(open_val) > 0.5)


def _preprocess_image(raw_img: np.ndarray) -> np.ndarray:
    """
    Flip 180° (matching training preprocessing from starVLA) and resize to IMAGE_SIZE.
    Input:  (H, W, 3) uint8
    Output: (IMAGE_SIZE, IMAGE_SIZE, 3) uint8
    """
    flipped = np.ascontiguousarray(raw_img[::-1, ::-1])
    resized = cv2.resize(flipped, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return resized


def _get_libero_env(task, resolution: int, seed: int):
    """Initialise a LIBERO OffScreenRenderEnv and return (env, task_description)."""
    task_description = task.language
    task_bddl_file = (
        pathlib.Path(_get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = _OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def eval_libero(args: Args) -> dict:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logging.info(f"Args: {json.dumps(dataclasses.asdict(args), indent=2)}")

    if args.task_suite_name not in MAX_STEPS:
        raise ValueError(
            f"Unknown task suite '{args.task_suite_name}'. "
            f"Choose from: {list(MAX_STEPS.keys())}"
        )
    max_steps = MAX_STEPS[args.task_suite_name]

    # Resolve norm_stats path (support relative-to-qwen_fast or absolute)
    norm_stats_path = pathlib.Path(args.norm_stats_path)
    if not norm_stats_path.is_absolute():
        norm_stats_path = pathlib.Path(__file__).resolve().parents[1] / norm_stats_path
    if not norm_stats_path.exists():
        raise FileNotFoundError(f"norm_stats not found at {norm_stats_path}")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    logging.info(f"Loading ContextVLAPolicy from {args.ckpt_path} ...")
    policy = ContextVLAPolicy(
        ckpt_path=args.ckpt_path,
        norm_stats_file_path=str(norm_stats_path),
        action_dim=args.action_dim,
        time_horizon=args.action_horizon,
        num_frames=args.num_frames,
        device=args.device,
    )
    logging.info("Model loaded.")

    # ------------------------------------------------------------------
    # Load LIBERO task suite
    # ------------------------------------------------------------------
    np.random.seed(args.seed)
    benchmark_dict = _libero_benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks = task_suite.n_tasks
    logging.info(f"Task suite '{args.task_suite_name}' — {num_tasks} tasks.")

    video_dir = pathlib.Path(args.video_out_path)
    video_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    total_episodes = 0
    total_successes = 0
    per_task_results = []

    black_frame = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

    for task_id in tqdm.tqdm(range(num_tasks), desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes = 0
        task_successes = 0

        for episode_idx in tqdm.tqdm(
            range(args.num_trials_per_task), desc=f"  Task {task_id}", leave=False
        ):
            # Reset image queues — pre-fill with black so queue is always full
            image_queue = deque(maxlen=args.num_frames)
            wrist_queue  = deque(maxlen=args.num_frames)
            for _ in range(args.num_frames):
                image_queue.append(black_frame.copy())
                wrist_queue.append(black_frame.copy())

            policy.reset()
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            replay_images = []
            t = 0
            done = False

            while t < max_steps + args.num_steps_wait:
                # Wait for objects to stabilise before sending real actions
                if t < args.num_steps_wait:
                    obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # ---- Preprocess images ----------------------------------------
                img       = _preprocess_image(obs["agentview_image"])
                wrist_img = _preprocess_image(obs["robot0_eye_in_hand_image"])

                image_queue.append(img)
                wrist_queue.append(wrist_img)

                if args.save_videos:
                    replay_images.append(img.copy())

                # ---- Build obs dict -------------------------------------------
                obs_dict = {
                    "image_queue":       np.stack(list(image_queue)),   # (num_frames, 224, 224, 3)
                    "wrist_image_queue": np.stack(list(wrist_queue)),   # (num_frames, 224, 224, 3)
                    "task_description":  task_description,
                }

                # ---- Inference ------------------------------------------------
                action = policy.infer(obs_dict)  # (action_horizon, action_dim)

                # Take the first predicted action from the chunk
                step_action = action[0]  # shape: (action_dim,) = (7,)

                # Parse action components
                world_vector  = step_action[:3]          # xyz delta
                rotation      = step_action[3:6]         # axis-angle delta
                gripper_cont  = float(step_action[6])    # continuous gripper
                gripper_bin   = _binarize_gripper(gripper_cont)  # {+1, -1}

                env_action = np.concatenate(
                    [world_vector, rotation, [gripper_bin]]
                ).tolist()  # length-7 list

                obs, reward, done, info = env.step(env_action)

                if done:
                    task_successes += 1
                    total_successes += 1
                    break

                t += 1

            task_episodes += 1
            total_episodes += 1

            # ---- Save video ---------------------------------------------------
            if args.save_videos and replay_images:
                suffix = "success" if done else "failure"
                safe_task = task_description.replace(" ", "_")[:60]
                video_path = (
                    video_dir
                    / f"task{task_id:02d}_ep{episode_idx:03d}_{suffix}.mp4"
                )
                imageio.mimwrite(
                    str(video_path),
                    [np.asarray(f) for f in replay_images],
                    fps=10,
                )

            logging.info(
                f"Task {task_id} | ep {episode_idx} | "
                f"{'SUCCESS' if done else 'FAILURE'} | "
                f"running total: {total_successes}/{total_episodes} "
                f"({100*total_successes/total_episodes:.1f}%)"
            )

        task_sr = task_successes / task_episodes if task_episodes > 0 else 0.0
        per_task_results.append({
            "task_id":          task_id,
            "task_description": task_description,
            "successes":        task_successes,
            "episodes":         task_episodes,
            "success_rate":     task_sr,
        })
        logging.info(
            f"Task {task_id} ({task_description}): "
            f"{task_successes}/{task_episodes} = {task_sr*100:.1f}%"
        )

        env.close()

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    overall_sr = total_successes / total_episodes if total_episodes > 0 else 0.0
    summary = {
        "ckpt_path":        args.ckpt_path,
        "task_suite":       args.task_suite_name,
        "num_frames":       args.num_frames,
        "total_episodes":   total_episodes,
        "total_successes":  total_successes,
        "overall_success_rate": overall_sr,
        "per_task":         per_task_results,
    }

    results_path = video_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    logging.info("=" * 60)
    logging.info(
        f"OVERALL SUCCESS RATE: {total_successes}/{total_episodes} = {overall_sr*100:.1f}%"
    )
    logging.info(f"Results saved to {results_path}")
    logging.info("=" * 60)

    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = tyro.cli(Args)
    eval_libero(args)
