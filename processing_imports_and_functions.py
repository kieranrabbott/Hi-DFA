import json
import random
import warnings

warnings.filterwarnings("ignore", message=".*cupyx.jit.rawkernel is experimental.*")

import numpy as np
import pandas as pd
import cv2
import skimage.measure
from skimage.measure import regionprops_table
import cupy as cp
from cupyx.scipy.signal import fftconvolve as gpu_fftconvolve
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import zarr
from numcodecs import Blosc
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import napari

compressor = Blosc(cname='zstd', clevel=9, shuffle=Blosc.BITSHUFFLE)

DEFAULT_MIN_SIZE = 100
DEFAULT_SOLIDITY = 0.86
min_size = DEFAULT_MIN_SIZE
min_solidity = DEFAULT_SOLIDITY

# Axis indices for 5D arrays (T, Z, C, Y, X)
C_AXIS = 2
Y_AXIS = 3
X_AXIS = 4


# ---------------------------------------------------------------------------
# Generic utilities & IO
# ---------------------------------------------------------------------------

def load_data(trenches_path, masks_path, config_path):
    """Loads trenches and masks for interactive viewing."""
    with open(config_path, "r") as f:
        config = json.load(f)
    channel_indices = config["channel_indices"]

    try:
        trenches_dask = da.from_zarr(trenches_path)
        masks_dask = da.from_zarr(masks_path)
        print(f"Loaded: {trenches_path}, {masks_path}")
    except Exception as e:
        raise FileNotFoundError(f"Could not load Zarr files: {e}")

    # Handle 5D (T, Z, C, Y, X) vs 4D inputs
    if trenches_dask.ndim == 5:
        pc_idx = channel_indices.get('PC', 1)
        trenches = trenches_dask[:, :, pc_idx, :, :]
        masks = masks_dask[:, :, 0, :, :]
    else:
        trenches, masks = trenches_dask, masks_dask
        
    return trenches, masks

def load_comparison_data(original_path, filtered_path, trenches_path, config_path):
    """Loads Original Masks, Filtered Masks, and Trenches for verification."""
    print(f"Comparison Viewer: Loading '{original_path}' vs '{filtered_path}'...")

    with open(config_path, "r") as f:
        config = json.load(f)
    channel_indices = config["channel_indices"]

    try:
        masks_orig = da.from_zarr(original_path)
        masks_filt = da.from_zarr(filtered_path)
        trenches_dask = da.from_zarr(trenches_path)
    except Exception as e:
        raise FileNotFoundError(f"Could not load Zarr files: {e}")

    # Handle 5D -> 4D for Trenches
    if trenches_dask.ndim == 5:
        pc_idx = channel_indices.get('PC', 1)
        trenches = trenches_dask[:, :, pc_idx, :, :]
    else:
        trenches = trenches_dask
        
    # Handle 5D -> 4D for Masks
    if masks_orig.ndim == 5: masks_orig = masks_orig[:, :, 0, :, :]
    if masks_filt.ndim == 5: masks_filt = masks_filt[:, :, 0, :, :]

    return trenches, masks_orig, masks_filt
    
def normalize_img(img):
    """Normalizes image intensity to 0-1 range based on percentiles."""
    img = img.astype(float)
    imin, imax = np.percentile(img, 1), np.percentile(img, 99)
    return np.clip((img - imin) / (imax - imin + 1e-8), 0, 1)

def get_overlay(img_gray, mask_binary, color=(1, 0, 0), alpha=0.3):
    """Creates an RGB overlay of a mask on a grayscale image."""
    img_rgb = np.stack([img_gray]*3, axis=-1)
    overlay = np.zeros_like(img_rgb)
    for i in range(3): overlay[..., i] = mask_binary * color[i]
    mask_indices = mask_binary > 0
    img_rgb[mask_indices] = alpha * overlay[mask_indices] + (1 - alpha) * img_rgb[mask_indices]
    return img_rgb


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def get_image_shifts(FL_image, masks, padding=8):
    """Calculates (x, y) shift using GPU FFT convolution."""
    FL_image_gpu = cp.asarray(FL_image)
    masks_gpu = cp.asarray(masks) > 0
    
    pad_width = ((padding, padding), (padding, padding))
    padded_masks = cp.pad(masks_gpu, pad_width, mode='constant')
    
    correlation = gpu_fftconvolve(padded_masks, FL_image_gpu[::-1, ::-1], mode='valid')
    y, x = cp.unravel_index(cp.argmax(correlation), correlation.shape)
    
    return int(x.get()) - padding, int(y.get()) - padding

def _transient_align_cycle(img_block, mask_block, out_dtype, channels=None, padding=8):
    """Upscales -> Aligns (skipping PC) -> Downscales."""
    t_dim, z_dim, c_dim = img_block.shape[:3]
    y_dim = img_block.shape[Y_AXIS]
    x_dim = img_block.shape[X_AXIS]
    
    img_final = np.zeros(img_block.shape, dtype=out_dtype)
    is_int = np.issubdtype(out_dtype, np.integer)
    mask_c_dim = mask_block.shape[C_AXIS]
    
    for i in range(t_dim):
        for j in range(z_dim):
            for c in range(c_dim):
                if channels and channels[c] == 'PC':
                    img_final[i, j, c] = img_block[i, j, c]
                    continue
                
                img_slice = img_block[i, j, c]
                mask_slice = mask_block[i, j, c if mask_c_dim > 1 else 0]
                
                img_high = cv2.resize(img_slice, (x_dim * 2, y_dim * 2), interpolation=cv2.INTER_CUBIC)
                
                if mask_slice.any():
                    dx, dy = get_image_shifts(img_high, mask_slice, padding=padding)
                    aligned = np.roll(img_high, shift=(dy, dx), axis=(0, 1))
                else:
                    aligned = img_high

                res = cv2.resize(aligned, (x_dim, y_dim), interpolation=cv2.INTER_AREA)
                img_final[i, j, c] = np.rint(res).astype(out_dtype) if is_int else res.astype(out_dtype)
    return img_final

def align_block_wrapper(img_block, mask_block, out_dtype, channels=None, padding=8):
    if mask_block.shape[Y_AXIS] == 2 * img_block.shape[Y_AXIS]:
        return _transient_align_cycle(img_block, mask_block, out_dtype, channels, padding)
    return img_block 

def prepare_masks(img_arr, mask_arr):
    m_y, m_x = mask_arr.shape[Y_AXIS], mask_arr.shape[X_AXIS]
    chunks = list(img_arr.chunks)
    chunks[C_AXIS] = (1,)
    chunks[Y_AXIS] = _scale_chunks(img_arr.chunks[Y_AXIS], img_arr.shape[Y_AXIS], m_y)
    chunks[X_AXIS] = _scale_chunks(img_arr.chunks[X_AXIS], img_arr.shape[X_AXIS], m_x)
    
    target_shape = list(img_arr.shape)
    target_shape[C_AXIS] = 1
    target_shape[Y_AXIS] = m_y
    target_shape[X_AXIS] = m_x
    return da.broadcast_to(mask_arr, tuple(target_shape)).rechunk(tuple(chunks))

def _scale_chunks(chunks, old_len, new_len):
    if old_len == new_len: return chunks
    scale = int(round(new_len / old_len))
    out = tuple(int(c * scale) for c in chunks)
    return out[:-1] + (out[-1] + (new_len - sum(out)),)

def run_alignment_test(images, masks, channels, padding=8, n_test_ids=3):
    """
    Selects random trenches, runs alignment, and visualizes the result.
    """
    import random
    
    # 1. Select Random IDs
    total = images.shape[0]
    # Ensure we don't request more samples than exist
    n_samples = min(n_test_ids, total)
    test_ids = sorted(random.sample(range(total), n_samples))
    
    print(f"--- Running Test Alignment on IDs: {test_ids} (Padding={padding}) ---")

    img_sub = images[test_ids]
    msk_sub = masks[test_ids]

    aligned_result = da.map_blocks(
        align_block_wrapper,
        img_sub,
        prepare_masks(img_sub, msk_sub),
        dtype=img_sub.dtype,
        chunks=img_sub.chunks,
        out_dtype=img_sub.dtype,
        channels=channels,
        padding=padding
    ).compute()
    
    print("Alignment Calculation Complete.")

    visualize_alignment(
        img_sub, 
        aligned_result, 
        msk_sub, 
        n_samples=n_samples, 
        trench_ids=test_ids
    )

def run_batch_alignment(images, masks, output_path, channels, padding=8, client=None):
    """
    Sets up the lazy alignment graph for the full dataset and writes it to Zarr.
    """
    print(f"--- Processing FULL Dataset (Padding={padding}) ---")

    aligned_full = da.map_blocks(
        align_block_wrapper,
        images,
        prepare_masks(images, masks),
        dtype=images.dtype,
        chunks=images.chunks,
        out_dtype=images.dtype,
        channels=channels,
        padding=padding
    )

    print("Writing to Zarr...")
    if client:
        print(f"Cluster: {client.dashboard_link}")

    aligned_full.to_zarr(output_path, compute=True, overwrite=True)
    
    print("Alignment Batch Complete.")


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

from dask.distributed import get_client
import time

def _remove_labels_fast(frame, bad_labels):
    """
    Removes specific labels from a frame using a Fast Lookup Table (LUT).
    """
    if not bad_labels: return frame
    
    max_lab = np.max(frame)
    if max_lab > 1_000_000: 
        frame[np.isin(frame, list(bad_labels))] = 0
        return frame

    lut = np.ones(max_lab + 1, dtype=frame.dtype)
    lut[list(bad_labels)] = 0
    frame[:] = lut[frame] * frame
    return frame

def _filter_frame_logic(frame, min_area, min_solidity, 
                        use_area, use_sol, use_edge, top_dist, side_dist):
    """Core filtering logic applied to a single 2D frame."""
    if not np.any(frame): return frame
    
    bad_labels = set()

    if use_edge:
        if top_dist > 0:
            bad_labels.update(np.unique(frame[:top_dist, :]))
        if side_dist > 0:
            bad_labels.update(np.unique(frame[:, :side_dist]))
            bad_labels.update(np.unique(frame[:, -side_dist:]))
        bad_labels.discard(0)

    if use_area or (use_sol and min_solidity > 0.0):
        labels, counts = np.unique(frame, return_counts=True)
        if use_area:
            small_lbls = labels[(labels != 0) & (counts < min_area)]
            bad_labels.update(small_lbls)

    if bad_labels:
        frame = _remove_labels_fast(frame, bad_labels)
        if not np.any(frame): return frame

    if use_sol and min_solidity > 0.0:
        from skimage.measure import regionprops_table
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            props = regionprops_table(frame, properties=('label', 'solidity'))
        
        if len(props['label']) > 0:
            sols = np.nan_to_num(props['solidity'], nan=1.0, posinf=1.0)
            solidity_bad = props['label'][sols < min_solidity]
            if len(solidity_bad) > 0:
                frame = _remove_labels_fast(frame, set(solidity_bad))
                
    return frame

def filter_chunk(block, min_area, min_solidity, 
                 use_area=True, use_sol=True, use_edge=False, 
                 top_dist=0, side_dist=0):
    """Called by Dask workers on individual chunks."""
    output = block.copy()
    if block.ndim == 3:
        for t in range(block.shape[0]):
            output[t] = _filter_frame_logic(output[t], min_area, min_solidity, 
                                            use_area, use_sol, use_edge, top_dist, side_dist)
    elif block.ndim == 4:
        for t in range(block.shape[0]):
            for z in range(block.shape[1]):
                output[t, z] = _filter_frame_logic(output[t, z], min_area, min_solidity, 
                                                   use_area, use_sol, use_edge, top_dist, side_dist)
    return output

def run_batch_filtering(input_path, output_path, filtering_params, n_jobs=20, limit=None):
    """
    Orchestrates parallel filtering.
    - limit: If integer provided, only processes the first N trenches (for benchmarking).
    """
    start_time = time.time()
    
    dask_in = da.from_zarr(input_path)
    source_z = zarr.open(input_path, mode='r')
    
    if limit is not None:
        dask_in = dask_in[:limit]
        print(f"** BENCHMARK MODE ** Processing first {limit} chunks only.")

    compressor = source_z.compressor
    
    print(f"Input Shape: {dask_in.shape} | Chunk Size: {source_z.chunks}")
    
    dask_filtered = dask_in.map_blocks(
        filter_chunk,
        min_area=filtering_params['min_area'],
        min_solidity=filtering_params['min_solidity'],
        use_area=filtering_params.get('use_area', True),
        use_sol=filtering_params.get('use_sol', True),
        use_edge=filtering_params.get('edge_filter', False),
        top_dist=filtering_params.get('top_limit', 0),
        side_dist=filtering_params.get('side_limit', 0),
        dtype=dask_in.dtype
    )
    
    z_out = zarr.open(
        output_path, mode='w', 
        shape=dask_in.shape, 
        chunks=source_z.chunks, 
        dtype=dask_in.dtype,
        compressor=compressor
    )
    
    try:
        client = get_client()
        print(f"Using Active Dask Cluster: {client.dashboard_link}")
        scheduler_kwargs = {} 
    except ValueError:
        print(f"No active cluster found. Using local 'processes' scheduler with {n_jobs} workers.")
        scheduler_kwargs = {'scheduler': 'processes', 'num_workers': n_jobs}

    dask_filtered.store(z_out, lock=False, **scheduler_kwargs)

    elapsed = time.time() - start_time
    print(f"Batch processing complete. Time taken: {elapsed/60:.2f} minutes.")
    


# ---------------------------------------------------------------------------
# Feature extraction (Dask)
# ---------------------------------------------------------------------------

def worker_extract_from_chunk(trench_block, mask_block, start_id, channels, channel_indices,
                              upscale=False, new_size=None, extract_features=None):
    """Worker: receives a Dask chunk (numpy array) and extracts features."""
    if mask_block.ndim == 5 and mask_block.shape[2] == 1:
        mask_block = mask_block[:, :, 0, :, :]

    n_batch = trench_block.shape[0]
    n_frames = trench_block.shape[1]
    batch_results = []

    for local_i in range(n_batch):
        global_id = start_id + local_i
        
        trench_data = trench_block[local_i]
        mask_data = mask_block[local_i]
        
        for t in range(n_frames):
            mask = mask_data[t]
            if not np.any(mask): continue
            
            # Prepare Intensity Images
            intensity_kwargs = {}
            valid_channels_map = []
            
            if extract_features and 'intensity' in extract_features:
                frame_img = trench_data[t]
                selected = []
                for ch in channels:
                    idx = channel_indices.get(ch)
                    if idx is not None and idx < frame_img.shape[0]:
                        selected.append(frame_img[idx])
                        valid_channels_map.append(ch)
                
                if selected:
                    stack = np.stack(selected, axis=-1)
                    if upscale and new_size:
                        stack = cv2.resize(stack, new_size, interpolation=cv2.INTER_CUBIC)
                    intensity_kwargs = {'intensity_image': stack}

            # Measure
            properties = ['label', 'centroid', 'area']
            if extract_features:
                if 'length' in extract_features: properties.append('major_axis_length')
                if 'width' in extract_features: properties.append('minor_axis_length')
                if 'solidity' in extract_features: properties.append('solidity')
                if 'intensity' in extract_features: properties.append('intensity_mean')

            try:
                props = regionprops_table(mask, properties=properties, **intensity_kwargs)
            except: continue
            
            if len(props['label']) == 0: continue
            
            # Select Mother Cell (lowest Y)
            min_idx = np.argmin(props['centroid-0'])
            
            row = {
                'mother_id': global_id,
                'timepoint': t,
                'area': props['area'][min_idx]
            }
            
            # Add optional features
            if extract_features:
                if 'length' in extract_features: row['length'] = props['major_axis_length'][min_idx]
                if 'width' in extract_features: row['width'] = props['minor_axis_length'][min_idx]
                if 'solidity' in extract_features: row['solidity'] = props['solidity'][min_idx]
            
            # Add intensity
            if extract_features and 'intensity' in extract_features and valid_channels_map:
                for i, ch_name in enumerate(valid_channels_map):
                    entry = row.copy()
                    val = None
                    k1 = f'intensity_mean-{i}'
                    if k1 in props: val = props[k1][min_idx]
                    elif 'intensity_mean' in props: val = props['intensity_mean'][min_idx]
                    
                    if val is not None:
                        entry['channel'] = ch_name
                        entry['intensity_raw'] = val
                        batch_results.append(entry)
            else:
                batch_results.append(row)
                
    return batch_results

def run_feature_extraction(trenches_path, masks_path, channels, channel_indices, 
                           extract_features, upscale=True):
    """
    Main Runner: Uses Dask Delayed to process Zarr chunks in parallel.
    Includes execution timer.
    """
    print(f"--- Starting Feature Extraction (Dask) ---")
    start_time = time.time()

    try:
        client = get_client()
        print(f"Using Cluster: {client.dashboard_link}")
    except ValueError:
        print("No active Dask client found. Creating local one...")
        cluster = LocalCluster()
        client = cluster.get_client()

    d_trenches = da.from_zarr(trenches_path)
    d_masks = da.from_zarr(masks_path)

    chunks_tr = d_trenches.to_delayed().ravel()
    chunks_mk = d_masks.to_delayed().ravel()
    
    chunk_sizes = d_trenches.chunks[0]
    
    mask_h, mask_w = d_masks.shape[-2:]
        
    new_size = (mask_w, mask_h) if upscale else None
    
    tasks = []
    current_id = 0
    
    for tr_delay, mk_delay, c_size in zip(chunks_tr, chunks_mk, chunk_sizes):
        
        task = dask.delayed(worker_extract_from_chunk)(
            tr_delay, mk_delay, 
            start_id=current_id,
            channels=channels, 
            channel_indices=channel_indices,
            upscale=upscale,
            new_size=new_size,
            extract_features=extract_features
        )
        tasks.append(task)
        current_id += c_size

    print(f"Generated {len(tasks)} tasks (Chunks). Computing...")

    futures = client.compute(tasks)
    dask.distributed.progress(futures)
    results_nested = client.gather(futures)
    
    print("\nAggregating results into DataFrame...")
    
    flat_results = [item for sublist in results_nested for item in sublist]
    
    df = pd.DataFrame()
    if flat_results:
        df = pd.DataFrame(flat_results)
        cols = ['mother_id', 'timepoint']
        if 'channel' in df.columns: cols.append('channel')
        if 'intensity_raw' in df.columns: cols.append('intensity_raw')
        remaining = [c for c in df.columns if c not in cols]
        df = df[cols + remaining]
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Extraction Complete. Time taken: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    return df
    


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

def convert_px_to_microns(df, pixel_size_um, upscale):
    """Converts pixel units to microns in place."""
    if df.attrs.get("spatial_units") == "um":
        print("Data is already in µm. Skipping conversion.")
        return df

    px_um = pixel_size_um / upscale
    linear_cols = [c for c in ['length', 'width'] if c in df.columns]
    area_cols   = [c for c in ['area'] if c in df.columns]

    if linear_cols: df[linear_cols] *= px_um
    if area_cols:   df[area_cols]   *= (px_um ** 2)

    if 'length' in df.columns: df["log_length"] = np.log(df["length"])
    if 'area' in df.columns:   df["log_area"]   = np.log(df["area"])

    df.attrs.update({"spatial_units": "um", "pixel_size": pixel_size_um, "upscale": upscale})
    print(f"Converted to µm (Scale: {px_um:.4f} µm/px)")
    return df

def reorder_dataframe(df, target_order):
    """
    Reorders DataFrame columns: places target columns first (if they exist), 
    followed by any remaining columns.
    """
    cols_present = [c for c in target_order if c in df.columns]
    cols_remaining = [c for c in df.columns if c not in cols_present]
    
    print("Columns reordered.")
    return df[cols_present + cols_remaining]

    

# ---------------------------------------------------------------------------
# Visualization (matplotlib)
# ---------------------------------------------------------------------------

def visualize_alignment(orig_data, aligned_data, mask_data, n_samples=3, trench_ids=None):
    """
    Visualizes alignment results. 
    Args:
        trench_ids (list, optional): List of real IDs corresponding to the data indices.
    """
    n_trenches, n_times, n_channels = orig_data.shape[:3]
    
    if n_samples < n_trenches:
        z_indices = random.sample(range(n_trenches), n_samples)
    else:
        z_indices = list(range(n_trenches))
        if n_samples > n_trenches:
             n_samples = n_trenches

    sample_indices = [(z, random.randint(0, n_times - 1)) for z in z_indices]

    print(f"Sampling {n_samples} locations...")
    
    lazy_raw = [orig_data[z, t] for z, t in sample_indices]
    lazy_aligned = [aligned_data[z, t] for z, t in sample_indices]
    lazy_masks = [mask_data[z, t, 0 if mask_data.ndim==5 else slice(None)] for z, t in sample_indices]
    
    if hasattr(orig_data, 'dask'):
        comp_raw, comp_aligned, comp_masks = da.compute(lazy_raw, lazy_aligned, lazy_masks)
    else:
        comp_raw, comp_aligned, comp_masks = lazy_raw, lazy_aligned, lazy_masks

    ref_h, ref_w = comp_masks[0].shape[-2:]
    aspect_ratio = ref_h / ref_w 
    plot_width = 2.0 
    plot_height = plot_width * aspect_ratio
    
    ncols = n_channels * 4
    nrows = n_samples
    total_w = ncols * plot_width
    total_h = nrows * plot_height
    
    fig, axes = plt.subplots(
        nrows, ncols, 
        figsize=(total_w, total_h), 
        gridspec_kw={'wspace': 0.05, 'hspace': 0.05}
    )
    
    if n_samples == 1: axes = axes[None, :] 

    for idx, (z, t) in enumerate(sample_indices):
        mask_h, mask_w = comp_masks[idx].shape[-2:]
        mask_bool = comp_masks[idx] > 0
        
        display_id = trench_ids[z] if trench_ids is not None else z

        for c in range(n_channels):
            col = c * 4
            
            img_orig = cv2.resize(normalize_img(comp_raw[idx][c]), (mask_w, mask_h), interpolation=cv2.INTER_NEAREST)
            img_algn = cv2.resize(normalize_img(comp_aligned[idx][c]), (mask_w, mask_h), interpolation=cv2.INTER_NEAREST)
            
            axes[idx, col].imshow(img_orig, cmap='gray', aspect='equal'); axes[idx, col].axis('off')
            axes[idx, col+1].imshow(get_overlay(img_orig, mask_bool), aspect='equal'); axes[idx, col+1].axis('off')
            axes[idx, col+2].imshow(img_algn, cmap='gray', aspect='equal'); axes[idx, col+2].axis('off')
            axes[idx, col+3].imshow(get_overlay(img_algn, mask_bool), aspect='equal'); axes[idx, col+3].axis('off')
            
            font_size = 12
            if idx == 0:
                axes[0, col].set_title(f"Ch{c}\nOriginal", fontsize=font_size)
                axes[0, col+1].set_title(f"Ch{c}\nOriginal+masks", fontsize=font_size)
                axes[0, col+2].set_title(f"Ch{c}\nAligned", fontsize=font_size)
                axes[0, col+3].set_title(f"Ch{c}\nAligned+masks", fontsize=font_size)
            
            if c == 0:
                lbl = f"ID {display_id}\nT {t}"
                axes[idx, 0].text(-0.2, 0.5, lbl, transform=axes[idx, 0].transAxes, 
                                  ha='right', va='center', fontsize=font_size, fontweight='bold')

    plt.show()

def visualize_background_logic(trenches_path, masks_path, channels, channel_indices, 
                               n_samples=3, bg_settings=None, upscale=True):
    """
    Visualizes background subtraction regions using a fixed region height.
    """
    # Default fallback
    bg_settings = bg_settings or {'buffer_from_cell': 10, 'region_height': 50, 'side_margin': 0}

    trenches = zarr.open(trenches_path, mode='r')
    masks = zarr.open(masks_path, mode='r')
    
    n_trenches, n_frames, _, H, W = trenches.shape
    
    sample_indices = []
    attempts = 0
    while len(sample_indices) < n_samples and attempts < 500:
        t, f = np.random.randint(0, n_trenches), np.random.randint(0, n_frames)
        if np.any(masks[t, f, 0]):
            sample_indices.append((t, f))
        attempts += 1

    if not sample_indices:
        print("Could not find frames with cells.")
        return

    ref_t, ref_f = sample_indices[0]
    ref_mask = masks[ref_t, ref_f, 0]
    ref_h, ref_w = ref_mask.shape
    aspect_ratio = ref_h / ref_w

    n_cols = len(channels) * 2
    n_rows = n_samples
    plot_width = 1.5 
    plot_height = plot_width * aspect_ratio
    total_w = n_cols * plot_width
    total_h = n_rows * plot_height

    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols, 
        figsize=(total_w, total_h), 
        gridspec_kw={'wspace': 0.05, 'hspace': 0.05},
        squeeze=False
    )
    
    fig.subplots_adjust(top=0.85, bottom=0.05, left=0.05, right=0.95)

    print(f"Checking Background Logic: {bg_settings}")

    for row_idx, (t_idx, f_idx) in enumerate(sample_indices):
        
        mask_frame = masks[t_idx, f_idx, 0]
        mask_h, mask_w = mask_frame.shape
        
        labeled_mask = skimage.measure.label(mask_frame)
        props = skimage.measure.regionprops(labeled_mask)
        
        if not props: continue 
        
        top_cell = min(props, key=lambda x: x.centroid[0])
        top_cell_mask = (labeled_mask == top_cell.label)
        
        cell_top_y = top_cell.bbox[0]
        bg_y_bottom = max(0, int(cell_top_y) - bg_settings['buffer_from_cell'])
        bg_y_top = max(0, bg_y_bottom - bg_settings['region_height'])
        
        bg_x_left = bg_settings.get('side_margin', 0)
        bg_x_right = mask_w - bg_settings.get('side_margin', 0)

        bg_mask = np.zeros_like(mask_frame, dtype=bool)
        bg_mask[bg_y_top:bg_y_bottom, bg_x_left:bg_x_right] = True

        raw_channels_data = trenches[t_idx, f_idx]

        for i, ch_name in enumerate(channels):
            col_raw = i * 2
            col_ovr = i * 2 + 1
            
            img = raw_channels_data[channel_indices[ch_name]]
            if upscale:
                img = cv2.resize(img, (mask_w, mask_h), interpolation=cv2.INTER_NEAREST)
            
            img_norm = normalize_img(img)
            img_rgb_raw = np.stack([img_norm]*3, axis=-1)
            
            overlay_layer = np.zeros_like(img_rgb_raw)
            overlay_layer[top_cell_mask, 0] = 1.0     
            overlay_layer[bg_mask, 1] = 0.5           
            overlay_layer[bg_mask, 2] = 1.0 

            any_mask = (top_cell_mask | bg_mask)
            img_rgb_overlay = img_rgb_raw.copy()
            alpha = 0.3
            img_rgb_overlay[any_mask] = (
                alpha * overlay_layer[any_mask] + 
                (1 - alpha) * img_rgb_overlay[any_mask]
            )

            ax_raw = axes[row_idx, col_raw]
            ax_ovr = axes[row_idx, col_ovr]
            
            ax_raw.imshow(img_rgb_raw, aspect='equal'); ax_raw.axis('off')
            ax_ovr.imshow(img_rgb_overlay, aspect='equal'); ax_ovr.axis('off')
            
            if row_idx == 0:
                ax_raw.set_title(f"{ch_name}", fontsize=10, fontweight='bold')
                ax_ovr.set_title(f"{ch_name} + overlay", fontsize=10, fontweight='bold')
            
            if i == 0:
                lbl = f"ID {t_idx}\nT {f_idx}"
                ax_raw.text(-0.2, 0.5, lbl, transform=ax_raw.transAxes, 
                            va='center', ha='right', fontsize=9, fontweight='bold')

    # Legend
    red_patch = mpatches.Patch(color='red', alpha=0.4, label='Top cell mask')
    blue_patch = mpatches.Patch(color='#0080FF', alpha=0.3, label='Background')
    
    fig.legend(handles=[red_patch, blue_patch], loc='lower center', 
               bbox_to_anchor=(0.5, 0.88), 
               ncol=2, fontsize=11, frameon=False)

    plt.show()

# Background extraction

def worker_bg_chunk(trench_block, mask_block, start_id, channels, channel_indices,
                    bg_settings, upscale=False, new_size=None):
    """Worker: extracts background intensity from a chunk of trenches."""
    if mask_block.ndim == 5 and mask_block.shape[2] == 1:
        mask_block = mask_block[:, :, 0, :, :]

    n_batch = trench_block.shape[0]
    n_frames = trench_block.shape[1]
    results = []

    buffer = bg_settings.get('buffer_from_cell', 10)
    height = bg_settings.get('region_height', 50)
    margin = bg_settings.get('side_margin', 0)

    for local_i in range(n_batch):
        global_id = start_id + local_i
        
        trench_data = trench_block[local_i]
        mask_data = mask_block[local_i]

        for t in range(n_frames):
            mask = mask_data[t]
            if not np.any(mask): continue

            rows, _ = np.where(mask > 0)
            if rows.size == 0: continue
            top_cell_y = np.min(rows)
            
            # Define Background Region (above cell)
            # (0,0) is top-left, so we subtract from top_cell_y to go up
            y_bot = max(0, top_cell_y - buffer)
            y_top = max(0, y_bot - height)
            
            # Check if region is valid
            if y_bot <= y_top: continue
            
            # Width limits
            x_L = margin
            x_R = mask.shape[1] - margin
            
            # Extract for each channel
            for ch in channels:
                idx = channel_indices.get(ch)
                if idx is None or idx >= trench_data.shape[1]: continue
                
                img = trench_data[t, idx]
                if upscale and new_size:
                    img = cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)
                
                bg_roi = img[y_top:y_bot, x_L:x_R]
                if bg_roi.size == 0: continue
                
                mean_val = np.mean(bg_roi)
                
                results.append({
                    'mother_id': global_id,
                    'timepoint': t,
                    'channel': ch,
                    'intensity_bg': mean_val
                })
                
    return results

def run_background_extraction(trenches_path, masks_path, channels, channel_indices, 
                              bg_settings, upscale=True):
    """
    Main Runner: Background extraction using Dask Delayed.
    """
    print("--- Starting Background Extraction (Dask) ---")
    start_time = time.time()
    
    try:
        client = get_client()
        print(f"Using Cluster: {client.dashboard_link}")
    except:
        cluster = LocalCluster()
        client = cluster.get_client()

    d_trenches = da.from_zarr(trenches_path)
    d_masks = da.from_zarr(masks_path)

    chunks_tr = d_trenches.to_delayed().ravel()
    chunks_mk = d_masks.to_delayed().ravel()
    chunk_sizes = d_trenches.chunks[0]

    mask_h, mask_w = d_masks.shape[-2:]
    new_size = (mask_w, mask_h) if upscale else None
    
    tasks = []
    current_id = 0
    
    for tr_d, mk_d, c_size in zip(chunks_tr, chunks_mk, chunk_sizes):
        task = dask.delayed(worker_bg_chunk)(
            tr_d, mk_d, current_id, channels, channel_indices, 
            bg_settings, upscale, new_size
        )
        tasks.append(task)
        current_id += c_size
        
    print(f"Generated {len(tasks)} tasks. Computing...")
    
    futures = client.compute(tasks)
    dask.distributed.progress(futures)
    results = client.gather(futures)

    flat = [item for sub in results for item in sub]
    df_bg = pd.DataFrame(flat)
    
    end_time = time.time()
    elapsed = end_time - start_time
    minutes, seconds = divmod(elapsed, 60)
    print(f"Background Extraction Complete. Time: {int(minutes)}m {seconds:.2f}s")
    
    return df_bg

def merge_background_data(df_main, df_bg):
    """
    Merges background data and calculates subtracted intensity.
    """
    if df_bg.empty:
        print("Warning: No background data to merge.")
        return df_main
        
    print("Merging background data...")
    # Drop old columns to avoid duplication/suffixes
    cols_to_drop = ['intensity_bg', 'intensity']
    df_main.drop(columns=[c for c in cols_to_drop if c in df_main.columns], inplace=True)
    
    # Merge
    merged = df_main.merge(df_bg, on=['mother_id', 'timepoint', 'channel'], how='left')
    
    # Subtract
    if 'intensity_raw' in merged.columns and 'intensity_bg' in merged.columns:
        merged['intensity'] = merged['intensity_raw'] - merged['intensity_bg']
        
    print(f"Merge Complete. New shape: {merged.shape}")
    return merged
    


# ---------------------------------------------------------------------------
# Interactive Napari filtering
# ---------------------------------------------------------------------------

class LazilyFilteredArray:
    """
    Wraps a Zarr array and applies filtering on-the-fly to requested slices.
    """
    def __init__(self, zarr_array, channel_idx=None, **params):
        self.source = zarr_array
        self.channel_idx = channel_idx 
        self.params = params
        
        if self.channel_idx is not None:
            self.shape = list(zarr_array.shape)
            self.shape.pop(2)
            self.shape = tuple(self.shape)
            self.ndim = zarr_array.ndim - 1
        else:
            self.shape = zarr_array.shape
            self.ndim = zarr_array.ndim
        self.dtype = zarr_array.dtype

    def update_params(self, **new_params):
        self.params.update(new_params)

    def __getitem__(self, key):
        if self.channel_idx is not None:
            if isinstance(key, tuple):
                real_key = list(key)
                real_key.insert(2, self.channel_idx)
                real_key = tuple(real_key)
            else:
                real_key = (key, slice(None), self.channel_idx, slice(None), slice(None))
        else:
            real_key = key

        raw_slice = self.source[real_key]
        return self._filter_in_memory(raw_slice)

    def _filter_in_memory(self, img):
        if not hasattr(img, 'ndim') or img.size == 0: return img

        if img.ndim == 2:
            return _filter_frame_logic(img, 
                self.params['min_area'], self.params['min_solidity'],
                self.params['use_area'], self.params['use_sol'], self.params['edge_filter'],
                self.params['top_limit'], self.params['side_limit'])
        
        out = np.empty_like(img)
        original_shape = img.shape
        flat_view = img.reshape(-1, original_shape[-2], original_shape[-1])
        out_flat = out.reshape(-1, original_shape[-2], original_shape[-1])
        
        for i in range(flat_view.shape[0]):
            out_flat[i] = _filter_frame_logic(flat_view[i], 
                self.params['min_area'], self.params['min_solidity'],
                self.params['use_area'], self.params['use_sol'], self.params['edge_filter'],
                self.params['top_limit'], self.params['side_limit'])
            
        return out

def run_interactive_filtering(
    trenches_path: str, 
    masks_path: str, 
    config_path: str, 
    full_config: dict, 
    upscale: bool = True
) -> dict:
    """Launches Napari to tune filtering parameters."""
    import napari
    from magicgui.widgets import Container, Slider, FloatSlider, PushButton, CheckBox, Label
    import sys

    try:
        trenches_zarr = zarr.open(trenches_path, mode='r')
        masks_zarr = zarr.open(masks_path, mode='r')
        with open(config_path, "r") as f: js = json.load(f)
        
        trenches_dask = da.from_zarr(trenches_path)
        channel_fix_masks = None
        
        if trenches_dask.ndim == 5:
            pc_idx = js["channel_indices"].get('PC', 1)
            trenches_display = trenches_dask[:, :, pc_idx, :, :]
            masks_source = masks_zarr
            channel_fix_masks = 0 
        else:
            trenches_display = trenches_dask
            masks_source = masks_zarr
            channel_fix_masks = None
    except Exception as e:
        print(f"Error loading data: {e}")
        return {}

    scale_img, scale_mask = ((1, 1, 2, 2), (1, 1, 1, 1)) if upscale else ((1, 1, 1, 1), (1, 1, 1, 1))
    width = trenches_display.shape[-1] * scale_img[-1]
    gap = 20
    trans_vec = (0, 0, 0, width + gap)

    viewer = napari.Viewer(title="Filter Tuning")
    viewer.add_image(trenches_display, name='Original', scale=scale_img, blending='additive')
    
    if channel_fix_masks is not None:
        masks_dask_display = da.from_zarr(masks_path)[:, :, channel_fix_masks, :, :]
    else:
        masks_dask_display = da.from_zarr(masks_path)
    viewer.add_labels(masks_dask_display, name='Masks (Raw)', scale=scale_mask)

    defaults = {
        'min_area': full_config['min_area']['default'],
        'min_solidity': full_config['solidity']['default'],
        'top_limit': full_config['top_limit']['default'],
        'side_limit': full_config['side_limit']['default'],
        'use_area': True,
        'use_sol': True,
        'edge_filter': True,
    }

    lazy_filtered_masks = LazilyFilteredArray(masks_source, channel_idx=channel_fix_masks, **defaults)

    viewer.add_image(trenches_display, name='Filtered', scale=scale_img, translate=trans_vec, blending='additive')
    layer_filtered = viewer.add_labels(lazy_filtered_masks, name='Masks (Filtered)', scale=scale_mask, translate=trans_vec)

    centers = [width / 2, width + gap + (width / 2)]
    viewer.add_points(np.array([[-20, x] for x in centers]), name='Labels', size=0, face_color='transparent', edge_color='transparent',
        text={'string': ['Original', 'Filtered'], 'color': 'white', 'size': 20, 'anchor': 'center'})
    
    results = defaults.copy()
    LABEL_WIDTH = 110
    lbl_area_main = Label(value="Area Filter:")
    lbl_area_main.min_width = LABEL_WIDTH
    chk_area = CheckBox(value=True, label="")
    lbl_area_sl = Label(value="Min Area (px):")
    sl_area = Slider(min=full_config['min_area']['min'], max=full_config['min_area']['max'], 
                     value=defaults['min_area'])
    
    row1 = Container(widgets=[lbl_area_main, chk_area, lbl_area_sl, sl_area], layout="horizontal", labels=False)

    lbl_sol_main = Label(value="Solidity Filter:")
    lbl_sol_main.min_width = LABEL_WIDTH
    chk_sol = CheckBox(value=True, label="")
    lbl_sol_sl = Label(value="Min Solidity:")
    sl_sol = FloatSlider(min=full_config['solidity']['min'], max=full_config['solidity']['max'], 
                         step=0.01, value=defaults['min_solidity'])
    
    row2 = Container(widgets=[lbl_sol_main, chk_sol, lbl_sol_sl, sl_sol], layout="horizontal", labels=False)

    lbl_edge_main = Label(value="Edge Filter:")
    lbl_edge_main.min_width = LABEL_WIDTH
    chk_edge = CheckBox(value=True, label="")
    
    lbl_top = Label(value="Top Limit (px):")
    sl_top = Slider(min=full_config['top_limit']['min'], max=full_config['top_limit']['max'], 
                    value=defaults['top_limit'])
    
    lbl_side = Label(value="Side Limit (px):")
    sl_side = Slider(min=full_config['side_limit']['min'], max=full_config['side_limit']['max'], 
                     value=defaults['side_limit'])

    row3 = Container(widgets=[lbl_edge_main, chk_edge, lbl_top, sl_top, lbl_side, sl_side], layout="horizontal", labels=False)

    w_save = PushButton(text="Finish and Save Parameters")
    w_save.native.setMinimumHeight(60)

    def update_view():
        params = {
            'use_area': chk_area.value, 'min_area': sl_area.value,
            'use_sol': chk_sol.value,   'min_solidity': sl_sol.value,
            'edge_filter': chk_edge.value, 
            'top_limit': sl_top.value, 'side_limit': sl_side.value
        }
        lazy_filtered_masks.update_params(**params)
        layer_filtered.refresh()
        
        status = []
        if params['use_area']: status.append(f"Area>{params['min_area']}")
        if params['use_sol']: status.append(f"Sol>{params['min_solidity']:.2f}")
        if params['edge_filter']: status.append(f"Edge(T{params['top_limit']}, S{params['side_limit']})")
        viewer.status = " | ".join(status) if status else "No Filters Active"

    for w in [chk_area, sl_area, chk_sol, sl_sol, chk_edge, sl_top, sl_side]:
        w.changed.connect(update_view)

    @w_save.clicked.connect
    def on_save():
        results.update({
            'use_area': chk_area.value, 'min_area': sl_area.value,
            'use_sol': chk_sol.value,   'min_solidity': sl_sol.value,
            'edge_filter': chk_edge.value, 
            'top_limit': sl_top.value, 'side_limit': sl_side.value
        })
        
        print("\n" + "="*40)
        print(" FILTERING PARAMETERS SAVED")
        print("="*40)
        if results['use_area']: print(f"Area Filter    : ON (> {results['min_area']} px)")
        else:                   print(f"Area Filter    : OFF")
            
        if results['use_sol']:  print(f"Solidity Filter: ON (> {results['min_solidity']:.2f})")
        else:                   print(f"Solidity Filter: OFF")
            
        if results['edge_filter']: 
            print(f"Edge Filter    : ON (Top: {results['top_limit']}, Side: {results['side_limit']})")
        else:
            print(f"Edge Filter    : OFF")
        print("="*40 + "\n")
        
        sys.stdout.flush()
        viewer.close()

    main_c = Container(widgets=[row1, row2, row3, w_save], layout="vertical", labels=False)
    viewer.window.add_dock_widget(main_c, area='bottom', name="Filter Controls")

    viewer.reset_view()
    viewer.dims.current_step = (0, 0, 0, 0)
    
    print("Napari running... Adjust sliders, then click 'Finish and Save Parameters'.")
    napari.run()
    
    return results

def run_comparison_viewer(trenches_path, masks_orig_path, masks_filt_path, config_path, upscale=True):
    """Opens original vs. filtered masks side-by-side in Napari."""
    print("Comparison Viewer: Opening...")

    trenches_dask = da.from_zarr(trenches_path)
    masks_dask = da.from_zarr(masks_orig_path)
    masks_filtered_dask = da.from_zarr(masks_filt_path)

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        PC_channel = config["channel_indices"].get("PC", 0)
    except:
        PC_channel = 0

    if trenches_dask.ndim == 5:
        trenches_channel_1 = trenches_dask[:, :, PC_channel, :, :]
    else:
        trenches_channel_1 = trenches_dask

    if masks_dask.ndim == 5:
        masks_view = masks_dask[:, :, 0, :, :]
    else:
        masks_view = masks_dask

    if masks_filtered_dask.ndim == 5:
        masks_view_filtered = masks_filtered_dask[:, :, 0, :, :]
    else:
        masks_view_filtered = masks_filtered_dask

    try:
        sample_frame = trenches_channel_1[0, 0].compute()
        clims = (sample_frame.min(), sample_frame.max())
    except:
        clims = None

    size_x = trenches_channel_1.shape[-1]
    
    if upscale:
        image_scale = (1, 1, 2, 2)
    else:
        image_scale = (1, 1, 1, 1)
        
    mask_scale = (1, 1, 1, 1)

    display_width = size_x * image_scale[-1]
    gap = 10 

    shift_amount = display_width + gap
    translation_vector = (0, 0, 0, shift_amount)

    viewer = napari.Viewer()

    viewer.add_image(
        trenches_channel_1,
        name='Original Image',
        scale=image_scale,
        blending='additive',
        contrast_limits=clims,
    )
    viewer.add_labels(masks_view, name='Masks (Original)', scale=mask_scale)

    viewer.add_image(
        trenches_channel_1,
        name='Filtered Image',
        scale=image_scale,
        translate=translation_vector,
        blending='additive',
        contrast_limits=clims,
    )

    viewer.add_labels(
        masks_view_filtered, 
        name='Masks (Filtered)', 
        scale=mask_scale,
        translate=translation_vector
    )

    # --- TEXT LABELS ---
    text_y_pos = -20
    center_left = display_width / 2
    center_right = display_width + gap + (display_width / 2)

    label_points = np.array([
        [text_y_pos, center_left],
        [text_y_pos, center_right]
    ])

    text_properties = {
        'string': ['Original', 'Filtered'],
        'color': 'white',
        'size': 20, 
        'anchor': 'center',
        'translation': np.array([0, 0])
    }

    viewer.add_points(
        label_points,
        name='Labels',
        size=0,
        text=text_properties,
        face_color='transparent',
        edge_color='transparent'
    )

    # --- STARTUP SETTINGS ---
    viewer.reset_view()
    
    # Force sliders to first frame to avoid "loading..." lag on startup
    ndim = trenches_channel_1.ndim
    viewer.dims.current_step = tuple([0] * ndim)

    napari.run()

    