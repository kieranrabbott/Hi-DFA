# omni_chunk76_utils.py

import os
import sys
import time
import json
import shutil
import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Tuple, List

import numpy as np
import cv2
import zarr
from joblib import Parallel, delayed
from tqdm import tqdm

from cellpose_omni import core


# -----------------------------------------------------------------------------
# Quiet / spam control (important for Jupyter stability)
# -----------------------------------------------------------------------------
_MAIN_PID = os.getpid()


def configure_quiet(silence_worker_prints: bool = True) -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message="IProgress not found.*")

    logging.getLogger().setLevel(logging.ERROR)
    for name in ("torch", "cellpose", "cellpose_omni", "omnipose", "zarr", "numcodecs", "tqdm"):
        logging.getLogger(name).setLevel(logging.ERROR)

    try:
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass

    if silence_worker_prints and os.getpid() != _MAIN_PID:
        devnull = open(os.devnull, "w")
        sys.stdout = devnull
        sys.stderr = devnull


# -----------------------------------------------------------------------------
# Per-process cached model
# -----------------------------------------------------------------------------
_MODEL_CACHE: Dict[Tuple[str, bool, int, int, int], Any] = {}


def get_model(
    model_name: str,
    use_gpu: bool,
    nchan: int = 1,
    nclasses: int = 2,
    dim: int = 2,
):
    key = (model_name, bool(use_gpu), int(nchan), int(nclasses), int(dim))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    from cellpose_omni import models

    model = models.CellposeModel(
        gpu=use_gpu,
        pretrained_model=model_name,
        nchan=nchan,
        nclasses=nclasses,
        dim=dim,
    )
    _MODEL_CACHE[key] = model
    return model


def safe_eval_concat(model, concat_img: np.ndarray, *, eval_kwargs: Dict[str, Any]) -> np.ndarray:
    import torch

    concat_img = np.nan_to_num(concat_img, copy=False)

    try:
        with torch.no_grad():
            masks, _, _ = model.eval([concat_img], **eval_kwargs)
        mask_concat = masks[0]
        if mask_concat.dtype != np.uint16:
            mask_concat = mask_concat.astype(np.uint16, copy=False)
        return mask_concat
    except Exception:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return np.zeros_like(concat_img, dtype=np.uint16)


def process_trench_chunk(
    t0: int,
    t1: int,
    *,
    input_path: str,
    output_path: str,
    lock_path: str,
    pc_index: int,
    model_name: str,
    use_gpu: bool,
    orig_h: int,
    orig_w: int,
    upscale: bool,
    eval_kwargs: Dict[str, Any],
    silence_worker_prints: bool = True,
) -> int:
    configure_quiet(silence_worker_prints=silence_worker_prints)

    inp = zarr.open(input_path, mode="r")
    synchronizer = zarr.ProcessSynchronizer(lock_path)
    out = zarr.open(output_path, mode="a", synchronizer=synchronizer)

    block = inp[t0:t1, :, pc_index, :, :]  # (B, T, H, W)
    B, T = block.shape[0], block.shape[1]

    model = get_model(model_name, use_gpu=use_gpu, nchan=1, nclasses=2, dim=2)

    out_h = orig_h * 2 if upscale else orig_h
    out_w = orig_w * 2 if upscale else orig_w

    out_block = np.empty((B, T, out_h, out_w), dtype=np.uint16)

    for bi in range(B):
        frames = block[bi]  # (T, H, W)

        # concat is (out_h, T*out_w)
        concat = np.empty((out_h, out_w * T), dtype=frames.dtype)
        for i in range(T):
            f = frames[i]
            if upscale:
                u = cv2.resize(f, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
            else:
                u = f  # no resize
            u = np.nan_to_num(u, copy=False)
            concat[:, i * out_w : (i + 1) * out_w] = u

        mask_concat = safe_eval_concat(model, concat, eval_kwargs=eval_kwargs)
        out_block[bi] = mask_concat.reshape(out_h, T, out_w).transpose(1, 0, 2)

    out[t0:t1, :, 0, :, :] = out_block
    return int(B * T)


# -----------------------------------------------------------------------------
# Public runner (keeps execution cell minimal; parameters still supplied by user)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class RunResult:
    frames_done: int
    seconds: float
    fps: float


def run_segmentation(
    *,
    input_zarr_path: str,
    output_zarr_path: str,
    lock_path: str,
    config_path: str,
    model_name: str,
    n_jobs: int,
    output_compressor: Any,
    eval_kwargs: Dict[str, Any],
    upscale: bool = True,
    silence_worker_prints: bool = True,
) -> RunResult:
    use_gpu = core.use_gpu()
    print(f">>> GPU activated? {use_gpu}")

    with open(config_path, "r") as f:
        config = json.load(f)
    pc_idx = config["channel_indices"]["PC"]

    inp = zarr.open(input_zarr_path, mode="r")
    num_trenches, num_frames = inp.shape[0], inp.shape[1]
    orig_h, orig_w = inp.shape[3], inp.shape[4]

    scale = 2 if upscale else 1
    out_h, out_w = orig_h * scale, orig_w * scale

    trench_block = int(inp.chunks[0])
    out_chunks = (trench_block, int(inp.chunks[1]), 1, out_h, out_w)

    print("Input:", input_zarr_path)
    print(" shape:", inp.shape, "dtype:", inp.dtype, "chunks:", inp.chunks)
    print(" upscale:", upscale, "| output (H,W):", (out_h, out_w))
    print(" trench_block (from input chunks[0]):", trench_block)
    print("Output:", output_zarr_path)
    print(" output shape:", (num_trenches, num_frames, 1, out_h, out_w))
    print(" output chunks:", out_chunks)

    # Clean output + lock
    if os.path.exists(output_zarr_path):
        shutil.rmtree(output_zarr_path, ignore_errors=True)
    if os.path.exists(lock_path):
        try:
            os.remove(lock_path)
        except Exception:
            pass

    synchronizer = zarr.ProcessSynchronizer(lock_path)

    # Create output store
    try:
        zarr.open(
            output_zarr_path,
            mode="w",
            shape=(num_trenches, num_frames, 1, out_h, out_w),
            chunks=out_chunks,
            dtype=np.uint16,
            compressor=output_compressor,
            synchronizer=synchronizer,
            dimension_separator="/",
        )
    except TypeError:
        zarr.open(
            output_zarr_path,
            mode="w",
            shape=(num_trenches, num_frames, 1, out_h, out_w),
            chunks=out_chunks,
            dtype=np.uint16,
            compressor=output_compressor,
            synchronizer=synchronizer,
        )

    blocks = [(t0, min(t0 + trench_block, num_trenches)) for t0 in range(0, num_trenches, trench_block)]

    t_start = time.time()

    parallel = Parallel(n_jobs=n_jobs, return_as="generator")
    jobs = (
        delayed(process_trench_chunk)(
            b0, b1,
            input_path=input_zarr_path,
            output_path=output_zarr_path,
            lock_path=lock_path,
            pc_index=pc_idx,
            model_name=model_name,
            use_gpu=use_gpu,
            orig_h=orig_h,
            orig_w=orig_w,
            upscale=upscale,
            eval_kwargs=eval_kwargs,
            silence_worker_prints=silence_worker_prints,
        )
        for (b0, b1) in blocks
    )

    frames_done = 0
    with tqdm(total=len(blocks), desc="chunk blocks") as pbar:
        for n_frames_done in parallel(jobs):
            frames_done += int(n_frames_done)
            pbar.update(1)

    seconds = time.time() - t_start
    fps = frames_done / seconds if seconds > 0 else 0.0
    return RunResult(frames_done=frames_done, seconds=seconds, fps=fps)