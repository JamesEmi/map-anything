# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
MapAnything Demo: Images-Only Inference with Visualization

Supports both MapAnything and external models (VGGT, DUSt3R, MASt3R, etc.)

Usage:
    # MapAnything (default)
    python demo_images_only_inference.py --image_folder /path/to/images --viz

    # External model (e.g., VGGT)
    python demo_images_only_inference.py --image_folder /path/to/images --external_model vggt --viz

    python demo_images_only_inference.py --help
"""

import argparse
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import matplotlib
import numpy as np
import rerun as rr
import torch

from mapanything.models import init_model, init_model_from_config, MapAnything
from mapanything.utils.geometry import (
    quaternion_to_rotation_matrix,
    recover_pinhole_intrinsics_from_ray_directions,
)
from mapanything.utils.image import load_images
from mapanything.utils.inference import (
    validate_input_views_for_inference,
    preprocess_input_views_for_inference,
    postprocess_model_outputs_for_inference,
)
from mapanything.utils.viz import (
    predictions_to_glb,
    script_add_rerun_args,
)

# Model configuration mapping: model_key -> (resolution, norm_type)
# Based on the table in README.md
EXTERNAL_MODEL_CONFIG = {
    "mapanything": (518, "dinov2"),
    "vggt": (518, "identity"),
    "dust3r": (512, "dust3r"),
    "mast3r": (512, "dust3r"),
    "must3r": (512, "dust3r"),
    "pi3": (518, "identity"),
    "pi3x": (518, "identity"),
    "pow3r": (512, "dust3r"),
    "pow3r_ba": (512, "dust3r"),
    "moge": (518, "identity"),
    "moge_1": (518, "identity"),
    "moge_2": (518, "identity"),
    "da3": (504, "dinov2"),
    "da3_nested": (504, "dinov2"),
}


def apply_colormap(
    values: np.ndarray,
    cmap_name: str = "turbo",
    vmin: float = None,
    vmax: float = None,
) -> np.ndarray:
    """Apply a colormap to values and return RGB image.

    Args:
        values: 2D array of values to colormap (H, W)
        cmap_name: Name of matplotlib colormap (e.g., 'turbo', 'viridis', 'plasma', 'inferno')
        vmin: Minimum value for normalization (None = use data min)
        vmax: Maximum value for normalization (None = use data max)

    Returns:
        RGB image as uint8 array (H, W, 3)
    """
    # Use provided bounds or fall back to data min/max
    v_min = vmin if vmin is not None else values.min()
    v_max = vmax if vmax is not None else values.max()

    # Normalize values to [0, 1]
    if v_max - v_min > 1e-8:
        normalized = (values - v_min) / (v_max - v_min)
        normalized = np.clip(normalized, 0, 1)  # Clip to handle out-of-range values
    else:
        normalized = np.zeros_like(values)

    # Apply colormap
    cmap = matplotlib.colormaps[cmap_name]
    colored = cmap(normalized)  # Returns (H, W, 4) RGBA

    # Convert to uint8 RGB
    rgb = (colored[:, :, :3] * 255).astype(np.uint8)
    return rgb


def log_data_to_rerun(
    image,
    depthmap,
    pose,
    intrinsics,
    pts3d,
    mask,
    base_name,
    pts_name,
    viz_mask=None,
    confidence=None,
    conf_vmin=None,
    conf_vmax=None,
    log_only_imgs_for_rerun_cams=False,
):
    """Log visualization data to Rerun"""
    # Log camera info and loaded data
    height, width = image.shape[0], image.shape[1]
    rr.log(
        base_name,
        rr.Transform3D(
            translation=pose[:3, 3],
            mat3x3=pose[:3, :3],
        ),
    )
    rr.log(
        f"{base_name}/pinhole",
        rr.Pinhole(
            image_from_camera=intrinsics,
            height=height,
            width=width,
            camera_xyz=rr.ViewCoordinates.RDF,
            image_plane_distance=1.0,
        ),
    )
    rr.log(
        f"{base_name}/pinhole/rgb",
        rr.Image(image),
    )
    if not log_only_imgs_for_rerun_cams:
        rr.log(
            f"{base_name}/pinhole/depth",
            rr.DepthImage(depthmap),
        )
        if viz_mask is not None:
            rr.log(
                f"{base_name}/pinhole/mask",
                rr.SegmentationImage(viz_mask.astype(int)),
            )
        if confidence is not None:
            # Apply colormap with absolute bounds for consistent visualization
            confidence_heatmap = apply_colormap(
                confidence, cmap_name="turbo", vmin=conf_vmin, vmax=conf_vmax
            )
            rr.log(
                f"{base_name}/pinhole/confidence",
                rr.Image(confidence_heatmap),
            )

    # Log points in 3D
    filtered_pts = pts3d[mask]
    filtered_pts_col = image[mask]

    rr.log(
        pts_name,
        rr.Points3D(
            positions=filtered_pts.reshape(-1, 3),
            colors=filtered_pts_col.reshape(-1, 3),
        ),
    )


def get_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="MapAnything Demo: Visualize metric 3D reconstruction from images"
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        required=True,
        help="Path to folder containing images for reconstruction",
    )
    parser.add_argument(
        "--apache",
        action="store_true",
        help="Use Apache 2.0 licensed model (facebook/map-anything-apache)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a local checkpoint file (overrides --apache if provided)",
    )
    parser.add_argument(
        "--external_model",
        type=str,
        default=None,
        choices=list(EXTERNAL_MODEL_CONFIG.keys()),
        help="Use an external model instead of MapAnything. "
        f"Available: {', '.join(EXTERNAL_MODEL_CONFIG.keys())}",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Override resolution for image loading (default: model-specific)",
    )
    parser.add_argument(
        "--norm_type",
        type=str,
        default=None,
        choices=["dinov2", "identity", "dust3r"],
        help="Override normalization type for image loading (default: model-specific)",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        default=False,
        help="Enable visualization with Rerun",
    )
    parser.add_argument(
        "--save_glb",
        action="store_true",
        default=False,
        help="Save reconstruction as GLB file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output.glb",
        help="Output path for GLB file (default: output.glb)",
    )
    parser.add_argument(
        "--video_viz_for_rerun",
        action="store_true",
        default=False,
        help="Video visualization for Rerun - logs views with time index",
    )
    parser.add_argument(
        "--log_only_imgs_for_rerun_cams",
        action="store_true",
        default=False,
        help="Log only images for Rerun camera - no depth, mask, etc.",
    )
    parser.add_argument(
        "--viz_confidence",
        action="store_true",
        default=False,
        help="Visualize predicted confidence scores in Rerun",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="Number of images per chunk. If None, process all images at once.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Number of overlapping images between consecutive chunks.",
    )
    parser.add_argument(
        "--inject_priors",
        action="store_true",
        default=False,
        help="Inject priors from previous chunk into overlap frames of current chunk.",
    )
    return parser


def load_mapanything_model(args, device):
    """Load MapAnything model from checkpoint or HuggingFace."""
    if args.checkpoint:
        # Load checkpoint and extract model args
        print(f"Loading checkpoint from: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

        # Initialize model using args from checkpoint
        if "args" in ckpt:
            model_args = ckpt["args"].model
            print(
                f"Initializing model from checkpoint args (model_str: {model_args.model_str})..."
            )
            model = init_model(model_args.model_str, model_args.model_config)
        else:
            # Fallback: use HuggingFace architecture if no args in checkpoint
            base_model_name = (
                "facebook/map-anything-apache"
                if args.apache
                else "facebook/map-anything"
            )
            print(
                f"No args in checkpoint, using HuggingFace architecture: {base_model_name}"
            )
            model = MapAnything.from_pretrained(base_model_name)

        model.to(device)

        # Load checkpoint weights
        if "model" in ckpt:
            load_result = model.load_state_dict(ckpt["model"], strict=False)
        else:
            load_result = model.load_state_dict(ckpt, strict=False)
        print(f"Checkpoint weights loaded: {load_result}")
        model.eval()
    elif args.apache:
        model_name = "facebook/map-anything-apache"
        print("Loading Apache 2.0 licensed MapAnything model from HuggingFace...")
        model = MapAnything.from_pretrained(model_name).to(device)
    else:
        model_name = "facebook/map-anything"
        print("Loading CC-BY-NC 4.0 licensed MapAnything model from HuggingFace...")
        model = MapAnything.from_pretrained(model_name).to(device)

    return model


def load_external_model(model_name, device):
    """Load external model using init_model_from_config."""
    print(f"Loading external model: {model_name}")
    model = init_model_from_config(model_name, device=device)
    model.eval()
    return model


def run_mapanything_inference(model, views):
    """Run inference with MapAnything model."""
    return model.infer(
        views,
        memory_efficient_inference=True,
        minibatch_size=1,
        use_amp=True,
        amp_dtype="bf16",
        apply_mask=True,
        mask_edges=True,
    )


def move_views_to_device(views, device):
    """Move all tensors in views to the specified device."""
    views_on_device = []
    for view in views:
        view_on_device = {}
        for key, value in view.items():
            if isinstance(value, torch.Tensor):
                view_on_device[key] = value.to(device)
            else:
                view_on_device[key] = value
        views_on_device.append(view_on_device)
    return views_on_device


def run_external_model_inference(model, views, device):
    """Run inference with external model using unified API."""
    # Move views to device (external models expect inputs on the correct device)
    views_on_device = move_views_to_device(views, device)
    with torch.no_grad():
        with torch.autocast(device):
            outputs = model(views_on_device)
    return outputs


def create_chunks(views, chunk_size, overlap):
    """
    Split views into chunks with specified overlap.

    Args:
        views: List of view dictionaries
        chunk_size: Number of views per chunk
        overlap: Number of overlapping views between chunks

    Returns:
        List of (chunk_views, overlap_indices, start_idx) tuples where:
        - chunk_views: List of view dicts for this chunk
        - overlap_indices: Indices within chunk that should receive priors (0 to overlap-1)
        - start_idx: Global index of the first view in this chunk
    """
    if chunk_size is None or chunk_size >= len(views):
        return [(views, [], 0)]

    chunks = []
    start = 0
    while start < len(views):
        end = min(start + chunk_size, len(views))
        chunk_views = views[start:end]

        # Determine which indices should receive priors (first `overlap` frames, except for first chunk)
        if start == 0:
            overlap_indices = []
        else:
            overlap_indices = list(range(min(overlap, len(chunk_views))))
            # overlap_indices = list(range(len(chunk_views)))

        chunks.append((chunk_views, overlap_indices, start))

        # Move start forward by (chunk_size - overlap)
        step = chunk_size - overlap
        if step <= 0:
            raise ValueError("overlap must be less than chunk_size")
        start += step

    return chunks


def inject_priors(views, overlap_indices, prev_outputs, device):
    """
    Inject geometric priors from previous chunk outputs into current chunk views.

    Uses native forward() format (depth_along_ray, ray_directions) following the
    adapter blueprint. This bypasses preprocessing conversion overhead.

    Args:
        views: List of view dictionaries to modify (will be modified in-place)
        overlap_indices: Indices within views that should receive priors
        prev_outputs: List of output dictionaries from previous chunk
        device: Torch device

    Returns:
        Modified views with priors injected in native forward() format
    """
    if not overlap_indices or prev_outputs is None:
        return views

    # Get the last `len(overlap_indices)` outputs from previous chunk
    num_overlap = len(overlap_indices)
    prior_outputs = prev_outputs[-num_overlap:]

    print(f"  Prior injection: overlap_indices={overlap_indices}, using {num_overlap} priors from prev chunk")
    for local_idx, prior_output in zip(overlap_indices, prior_outputs):
        view = views[local_idx]

        # Extract priors in native forward() format
        depth_along_ray = prior_output["depth_along_ray"].clone()  # (B, H, W, 1)
        ray_directions = prior_output["ray_directions"].clone()  # (B, H, W, 3)

        # Get mask and zero out invalid depths (model treats depth <= 0 as invalid)
        if "mask" in prior_output and prior_output["mask"] is not None:
            mask = prior_output["mask"]  # (B, H, W, 1)
            depth_along_ray[~mask] = 0.0

        # Inject in native forward() format
        view["depth_along_ray"] = depth_along_ray.to(device)
        view["ray_directions"] = ray_directions.to(device)

        # Remove conflicting keys that would cause preprocessing to convert them
        if "intrinsics" in view:
            del view["intrinsics"]

    return views


def run_chunked_inference(
    model,
    views,
    chunk_size,
    overlap,
    inject_priors_flag,
    device,
    use_external_model,
    memory_efficient_inference=True,
    minibatch_size=1,
    use_amp=True,
    amp_dtype="bf16",
):
    """
    Run inference on views in chunks with optional prior injection.

    Args:
        model: The MapAnything model
        views: List of all view dictionaries
        chunk_size: Number of views per chunk
        overlap: Number of overlapping views between chunks
        inject_priors_flag: Whether to inject priors from previous chunk
        device: Torch device
        use_external_model: Whether using external model (different inference path)
        memory_efficient_inference: Use memory efficient mode for forward()
        minibatch_size: Minibatch size for memory efficient inference
        use_amp: Use automatic mixed precision
        amp_dtype: AMP dtype ("bf16" or "fp16")

    Returns:
        List of all output dictionaries (one per view, de-duplicated)
    """
    # Create chunks
    chunks = create_chunks(views, chunk_size, overlap)

    prev_chunk_outputs = None
    global_view_outputs = {}  # Track outputs by global index to handle overlaps

    for chunk_idx, (chunk_views, overlap_indices, start_idx) in enumerate(chunks):
        print(
            f"Processing chunk {chunk_idx + 1}/{len(chunks)} "
            f"(views {start_idx} to {start_idx + len(chunk_views) - 1})"
        )

        # Deep copy views to avoid modifying originals
        chunk_views_copy = [{k: v for k, v in v.items()} for v in chunk_views]

        # Move views to device first
        chunk_views_on_device = move_views_to_device(chunk_views_copy, device)

        # Run inference based on model type
        if use_external_model:
            # Inject priors if enabled (external models - not supported)
            if inject_priors_flag and prev_chunk_outputs is not None and overlap_indices:
                print("  Warning: Prior injection not supported for external models, skipping")
            chunk_outputs = run_external_model_inference(model, chunk_views_on_device, device)
        else:
            # For MapAnything, use forward() directly with preprocessing
            # Validate first
            validated_views = validate_input_views_for_inference(chunk_views_on_device)

            # Inject priors AFTER validation but BEFORE preprocessing (native forward() format)
            if inject_priors_flag and prev_chunk_outputs is not None and overlap_indices:
                print(f"  Injecting priors into {len(overlap_indices)} overlap frames (native forward() format)")
                validated_views = inject_priors(
                    validated_views,
                    overlap_indices,
                    prev_chunk_outputs,
                    device,
                )

            # Preprocess (converts intrinsics -> ray_directions for non-prior views)
            processed_views = preprocess_input_views_for_inference(validated_views)

            # Run forward pass
            amp_dtype_torch = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
            with torch.no_grad():
                with torch.autocast(device, enabled=use_amp, dtype=amp_dtype_torch):
                    raw_outputs = model.forward(
                        processed_views,
                        memory_efficient_inference=memory_efficient_inference,
                        minibatch_size=minibatch_size,
                    )

            # Post-process outputs
            chunk_outputs = postprocess_model_outputs_for_inference(
                raw_outputs=raw_outputs,
                input_views=processed_views,
                apply_mask=True,
                mask_edges=False, # False because we don't want depths set to 0 - OOD.
            )

        # Print mean depth for this chunk
        chunk_depths = []
        for output in chunk_outputs:
            if "depth_z" in output:
                depth = output["depth_z"][0].cpu().numpy()
                mask = output["mask"][0].squeeze(-1).cpu().numpy() if "mask" in output else (depth > 0)
                if mask.any():
                    chunk_depths.append(depth[mask].mean())
        if chunk_depths:
            print(f"  Chunk {chunk_idx + 1} mean depth: {np.mean(chunk_depths):.4f}")

        # Store outputs by global index (use latest for overlaps)
        for local_idx, output in enumerate(chunk_outputs):
            global_idx = start_idx + local_idx
            global_view_outputs[global_idx] = output

        # Save outputs for next chunk's priors
        prev_chunk_outputs = chunk_outputs

    # Convert to ordered list
    all_outputs = [global_view_outputs[i] for i in sorted(global_view_outputs.keys())]

    return all_outputs


def main():
    # Parser for arguments and Rerun
    parser = get_parser()
    script_add_rerun_args(
        parser
    )  # Options: --headless, --connect, --serve, --addr, --save, --stdout
    args = parser.parse_args()

    # Get inference device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Determine if using external model
    use_external_model = args.external_model is not None

    # Determine resolution and norm_type
    if use_external_model:
        model_key = args.external_model
        default_resolution, default_norm_type = EXTERNAL_MODEL_CONFIG[model_key]
    else:
        # MapAnything defaults
        default_resolution, default_norm_type = 518, "dinov2"

    resolution = args.resolution if args.resolution is not None else default_resolution
    norm_type = args.norm_type if args.norm_type is not None else default_norm_type

    # Initialize model
    if use_external_model:
        model = load_external_model(args.external_model, device)
        model_display_name = args.external_model.upper()
    else:
        model = load_mapanything_model(args, device)
        model_display_name = "MapAnything"

    # Load images with appropriate settings
    # Use fixed_size to ensure output resolution matches input for prior injection
    print(f"Loading images from: {args.image_folder}")
    print(f"  Resolution: {resolution}, Normalization: {norm_type}")
    # stored_h, stored_w = prior_predictions.depth_along_ray.shape[1:3]
    views = load_images(
        args.image_folder,
        resolution_set=resolution,
        norm_type=norm_type,
        patch_size=14,
        resize_mode="fixed_size",
        size=(294, 294) #TODO: make this flexible, pull from prior_pred.depth_along_ray.shape 
    )
    print(f"Loaded {len(views)} views")

    # Validate chunking parameters
    if args.chunk_size is not None:
        if args.chunk_size < 2:
            raise ValueError("chunk_size must be at least 2")
        if args.overlap >= args.chunk_size:
            raise ValueError("overlap must be less than chunk_size")
        if args.overlap < 0:
            raise ValueError("overlap must be non-negative")

    # Run model inference
    print(f"Running {model_display_name} inference...")
    if args.chunk_size is not None:
        print(f"  Chunked inference: chunk_size={args.chunk_size}, overlap={args.overlap}, inject_priors={args.inject_priors}")
        outputs = run_chunked_inference(
            model=model,
            views=views,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            inject_priors_flag=args.inject_priors,
            device=device,
            use_external_model=use_external_model,
        )
    else:
        # Original single-batch inference
        if use_external_model:
            outputs = run_external_model_inference(model, views, device)
        else:
            outputs = run_mapanything_inference(model, views)
    print("Inference complete!")

    # Prepare lists for GLB export if needed
    world_points_list = []
    images_list = []
    masks_list = []

    # Compute global confidence range for absolute colormap scaling
    conf_min, conf_max = None, None
    if args.viz and args.viz_confidence:
        all_conf_values = []
        for pred in outputs:
            if "conf" in pred and pred["conf"] is not None:
                conf = pred["conf"][0].cpu().numpy()
                all_conf_values.append(conf)
        if all_conf_values:
            all_conf_concat = np.concatenate([c.flatten() for c in all_conf_values])
            conf_min, conf_max = float(all_conf_concat.min()), float(all_conf_concat.max())
            print(f"Global confidence range: [{conf_min:.4f}, {conf_max:.4f}]")

    # Initialize Rerun if visualization is enabled
    if args.viz:
        print("Starting visualization...")
        if args.video_viz_for_rerun:
            viz_string = f"{model_display_name}_Video_Visualization"
        else:
            viz_string = f"{model_display_name}_Visualization"
        rr.script_setup(args, viz_string)
        rr.set_time("stable_time", sequence=0)
        rr.log("reconstruction", rr.ViewCoordinates.RDF, static=True)

    # Initialize accumulators for visualization
    all_pts = []  # Accumulates across ALL chunks for final pointcloud
    all_cols = []
    chunk_pts = []  # Accumulates within current chunk
    chunk_cols = []

    # Compute chunk boundaries for visualization
    if args.chunk_size is not None:
        chunks_info = create_chunks(views, args.chunk_size, args.overlap)
        chunk_end_indices = {start_idx + len(chunk_views) - 1 for chunk_views, _, start_idx in chunks_info}
    else:
        chunks_info = []
        chunk_end_indices = set()

    time_idx = 0  # Track time sequence for Rerun

    # Loop through the outputs
    for view_idx, pred in enumerate(outputs):
        # Extract data from predictions - unified output format
        # Handle both MapAnything and external model outputs

        # Get intrinsics first (needed for depth conversion)
        if "intrinsics" in pred:
            intrinsics_torch = pred["intrinsics"][0]  # (3, 3)
        elif "ray_directions" in pred:
            # Compute intrinsics from ray directions using existing utility
            intrinsics_torch = recover_pinhole_intrinsics_from_ray_directions(
                pred["ray_directions"][0]
            )
        else:
            intrinsics_torch = None

        # Get z-depth for Rerun visualization (DepthImage expects z-depth, not depth-along-ray)
        if "depth_z" in pred:
            depthmap_torch = pred["depth_z"][0].squeeze(-1)  # (H, W)
        elif "depth_along_ray" in pred and "ray_directions" in pred:
            # Convert depth_along_ray to depth_z: depth_z = depth_along_ray * ray_dir_z
            depth_along_ray = pred["depth_along_ray"][0].squeeze(-1)  # (H, W)
            ray_dirs = pred["ray_directions"][0]  # (H, W, 3)
            ray_dir_z = ray_dirs[:, :, 2]  # z-component of normalized ray directions
            depthmap_torch = depth_along_ray * ray_dir_z
        elif "pts3d_cam" in pred:
            # Compute z-depth from pts3d_cam
            depthmap_torch = pred["pts3d_cam"][0, :, :, 2]  # Z component
        else:
            depthmap_torch = None

        # Get camera pose - construct from cam_trans and cam_quats if needed
        if "camera_poses" in pred:
            camera_pose_torch = pred["camera_poses"][0]  # (4, 4)
        elif "cam_trans" in pred and "cam_quats" in pred:
            cam_trans = pred["cam_trans"][0]  # (3,)
            cam_quats = pred["cam_quats"][0]  # (4,)
            rot_mat = quaternion_to_rotation_matrix(cam_quats.unsqueeze(0))[0]  # (3, 3)
            camera_pose_torch = torch.eye(4, device=cam_trans.device)
            camera_pose_torch[:3, :3] = rot_mat
            camera_pose_torch[:3, 3] = cam_trans
        else:
            camera_pose_torch = torch.eye(4, device=device)

        # Get pts3d - prefer world coordinates if available
        if "pts3d" in pred:
            pts3d_torch = pred["pts3d"][0]
        elif "pts3d_cam" in pred and camera_pose_torch is not None:
            # Transform camera points to world frame
            pts3d_cam = pred["pts3d_cam"][0]  # (H, W, 3)
            H, W, _ = pts3d_cam.shape
            pts_flat = pts3d_cam.reshape(-1, 3)  # (H*W, 3)
            pts_homo = torch.cat([pts_flat, torch.ones(pts_flat.shape[0], 1, device=pts_flat.device)], dim=1)
            pts_world = (camera_pose_torch @ pts_homo.T).T[:, :3]
            pts3d_torch = pts_world.reshape(H, W, 3)
        else:
            pts3d_torch = None

        # Get or create mask
        if "mask" in pred and pred["mask"] is not None:
            mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
        else:
            # Create mask from valid depth values
            if depthmap_torch is not None:
                mask = (depthmap_torch > 0).cpu().numpy()
            elif pts3d_torch is not None:
                mask = torch.isfinite(pts3d_torch).all(dim=-1).cpu().numpy()
            else:
                mask = np.ones((views[0]["img"].shape[2], views[0]["img"].shape[3]), dtype=bool)

        # Get image for visualization
        if "img_no_norm" in pred:
            image_np = pred["img_no_norm"][0].cpu().numpy()
        else:
            # Denormalize from views
            img_tensor = views[view_idx]["img"][0]  # (3, H, W)
            if norm_type == "dinov2":
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_denorm = img_tensor.cpu() * std + mean
            elif norm_type == "dust3r":
                img_denorm = (img_tensor.cpu() + 1.0) / 2.0
            else:  # identity
                img_denorm = img_tensor.cpu()
            image_np = (img_denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # Convert tensors to numpy
        pts3d_np = pts3d_torch.cpu().numpy() if pts3d_torch is not None else None
        depthmap_np = depthmap_torch.cpu().numpy() if depthmap_torch is not None else None
        intrinsics_np = intrinsics_torch.cpu().numpy() if intrinsics_torch is not None else np.eye(3)
        camera_pose_np = camera_pose_torch.cpu().numpy()

        # Store data for GLB export if needed
        if args.save_glb and pts3d_np is not None:
            world_points_list.append(pts3d_np)
            images_list.append(image_np)
            masks_list.append(mask)

        # Accumulate points for visualization
        if pts3d_np is not None and mask is not None:
            pts_now = pts3d_np[mask].reshape(-1, 3)
            if image_np.max() <= 1.0:
                cols_now = (image_np[mask].reshape(-1, 3) * 255).astype(np.uint8)
            else:
                cols_now = image_np[mask].reshape(-1, 3).astype(np.uint8)
            # Accumulate for chunk and overall
            chunk_pts.append(pts_now)
            chunk_cols.append(cols_now)
            all_pts.append(pts_now)
            all_cols.append(cols_now)

        # Log to Rerun if visualization is enabled
        if args.viz and pts3d_np is not None:
            rr.set_time("stable_time", sequence=time_idx)
            time_idx += 1

            if args.video_viz_for_rerun:
                view_base_name = "reconstruction/view"
            else:
                view_base_name = f"reconstruction/view_{view_idx}"

            # Extract confidence if visualization is enabled
            conf_np = None
            if args.viz_confidence and "conf" in pred and pred["conf"] is not None:
                conf_np = pred["conf"][0].cpu().numpy()  # (H, W)

            log_data_to_rerun(
                image=image_np,
                depthmap=depthmap_np if depthmap_np is not None else np.zeros_like(mask, dtype=np.float32),
                pose=camera_pose_np,
                intrinsics=intrinsics_np,
                pts3d=pts3d_np,
                mask=mask,
                base_name=view_base_name,
                pts_name=f"reconstruction/pointcloud_view_{view_idx}",
                viz_mask=mask,
                confidence=conf_np,
                conf_vmin=conf_min,
                conf_vmax=conf_max,
                log_only_imgs_for_rerun_cams=args.log_only_imgs_for_rerun_cams,
            )

            # At end of chunk, log chunk's accumulated pointcloud for ~2 time steps
            if view_idx in chunk_end_indices and chunk_pts:
                chunk_pts_combined = np.concatenate(chunk_pts, axis=0)
                chunk_cols_combined = np.concatenate(chunk_cols, axis=0)

                # Log chunk pointcloud for 2 time steps
                for _ in range(2):
                    rr.set_time("stable_time", sequence=time_idx)
                    time_idx += 1
                    rr.log(
                        "reconstruction/chunk_pointcloud",
                        rr.Points3D(positions=chunk_pts_combined, colors=chunk_cols_combined),
                    )

                # Reset chunk accumulators for next chunk
                chunk_pts = []
                chunk_cols = []

    if args.viz:
        # Log final accumulated pointcloud
        if all_pts:
            print("Logging final accumulated point cloud...")
            final_pts = np.concatenate(all_pts, axis=0)
            final_cols = np.concatenate(all_cols, axis=0)

            # Subsample if too large
            max_points = 2000000
            if final_pts.shape[0] > max_points:
                sel = np.random.choice(final_pts.shape[0], size=max_points, replace=False)
                final_pts = final_pts[sel]
                final_cols = final_cols[sel]

            rr.set_time("stable_time", sequence=time_idx)
            rr.log(
                "reconstruction/final_pointcloud",
                rr.Points3D(positions=final_pts, colors=final_cols),
            )
            print(f"Final point cloud: {final_pts.shape[0]:,} points")

        print("Visualization complete! Check the Rerun viewer.")

    # Export GLB if requested
    if args.save_glb:
        if world_points_list:
            print(f"Saving GLB file to: {args.output_path}")

            # Stack all views
            world_points = np.stack(world_points_list, axis=0)
            images = np.stack(images_list, axis=0)
            final_masks = np.stack(masks_list, axis=0)

            # Create predictions dict for GLB export
            predictions = {
                "world_points": world_points,
                "images": images,
                "final_masks": final_masks,
            }

            # Convert to GLB scene
            scene_3d = predictions_to_glb(predictions, as_mesh=True)

            # Save GLB file
            scene_3d.export(args.output_path)
            print(f"Successfully saved GLB file: {args.output_path}")
        else:
            print("No valid 3D points to export to GLB")
    else:
        print("Skipping GLB export (--save_glb not specified)")


if __name__ == "__main__":
    main()