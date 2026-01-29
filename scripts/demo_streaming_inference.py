# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
MapAnything Demo: Streaming Inference with Chunked Processing

Processes image sequences in chunks with sliding window. Overlapping predictions
from previous chunks are provided as input to subsequent chunks.

Looped inference (stability testing) is a subset: set window_size >= N and num_iterations > 1.

Usage:
    python demo_streaming_inference.py --help
"""

import argparse
import os
import warnings

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import rerun as rr
import torch

from mapanything.models import MapAnything
from mapanything.utils.geometry import (
    depth_edge,
    normals_edge,
    points_to_normals,
    quaternion_to_rotation_matrix,
    recover_pinhole_intrinsics_from_ray_directions,
    rotation_matrix_to_quaternion,
)
from mapanything.utils.image import load_images, rgb
from mapanything.utils.viz import script_add_rerun_args
from mapanything.utils.alignment.alignment_torch import robust_weighted_estimate_sim3_torch


def compute_viz_mask(pts3d_np, depth_z_np, non_ambiguous_mask_np):
    """Compute visualization mask with edge masking."""
    if not non_ambiguous_mask_np.any():
        return non_ambiguous_mask_np

    normals, normals_mask = points_to_normals(pts3d_np, mask=non_ambiguous_mask_np)
    normal_edges = normals_edge(normals, tol=5.0, mask=normals_mask)
    depth_edges = depth_edge(depth_z_np, rtol=0.03, mask=non_ambiguous_mask_np)
    edge_mask = ~(depth_edges & normal_edges)
    return non_ambiguous_mask_np & edge_mask


def compute_debug_stats(prev_pred, curr_pred, view_idx, iteration):
    """Compute and print debug statistics comparing previous vs current predictions."""
    # Get common valid mask
    prev_mask = prev_pred["non_ambiguous_mask"][0].cpu().numpy()
    curr_mask = curr_pred["non_ambiguous_mask"][0].cpu().numpy()
    common_mask = prev_mask & curr_mask

    if not common_mask.any():
        print(f"  [DEBUG] View {view_idx}: No common valid pixels")
        return

    # Depth comparison (input vs output)
    prev_depth = prev_pred["depth_along_ray"][0, :, :, 0].cpu().numpy()
    curr_depth = curr_pred["depth_along_ray"][0, :, :, 0].cpu().numpy()
    depth_diff = np.abs(curr_depth - prev_depth)[common_mask]
    depth_rel_diff = depth_diff / (prev_depth[common_mask] + 1e-6)

    # Ray directions comparison (angular difference in degrees)
    prev_rays = prev_pred["ray_directions"][0].cpu().numpy()  # (H, W, 3)
    curr_rays = curr_pred["ray_directions"][0].cpu().numpy()
    # Normalize rays
    prev_rays_norm = prev_rays / (np.linalg.norm(prev_rays, axis=-1, keepdims=True) + 1e-8)
    curr_rays_norm = curr_rays / (np.linalg.norm(curr_rays, axis=-1, keepdims=True) + 1e-8)
    # Compute angular difference
    ray_dot = np.sum(prev_rays_norm * curr_rays_norm, axis=-1)
    ray_angle_diff = np.arccos(np.clip(ray_dot, -1, 1)) * 180 / np.pi  # degrees
    ray_angle_diff_valid = ray_angle_diff[common_mask]

    # Pose comparison
    prev_trans = prev_pred["cam_trans"][0].cpu().numpy()
    curr_trans = curr_pred["cam_trans"][0].cpu().numpy()
    trans_diff = np.linalg.norm(curr_trans - prev_trans)

    prev_quats = prev_pred["cam_quats"][0].cpu().numpy()
    curr_quats = curr_pred["cam_quats"][0].cpu().numpy()
    # Quaternion distance (angle in degrees)
    dot = np.abs(np.dot(prev_quats, curr_quats))
    angle_diff = 2 * np.arccos(np.clip(dot, -1, 1)) * 180 / np.pi

    print(f"  [DEBUG] View {view_idx} (iter {iteration}):")
    print(f"    Depth - abs diff: mean={depth_diff.mean():.4f}, max={depth_diff.max():.4f}")
    print(f"    Depth - rel diff: mean={depth_rel_diff.mean():.4f}, max={depth_rel_diff.max():.4f}")
    print(f"    Rays  - angle diff (deg): mean={ray_angle_diff_valid.mean():.4f}, max={ray_angle_diff_valid.max():.4f}")
    print(f"    Pose  - trans diff: {trans_diff:.6f}, rot diff: {angle_diff:.4f} deg")
    print(f"    Mask  - prev: {prev_mask.sum()}, curr: {curr_mask.sum()}, common: {common_mask.sum()}")


def log_to_rerun(image_np, depth_z_np, pose_np, intrinsics_np, pts3d_np, viz_mask, base_name, pts_name,
                 log_cameras=True, log_depth_mask=False):
    """Log visualization data to Rerun."""
    # Always log pointcloud
    rr.log(pts_name, rr.Points3D(
        positions=pts3d_np[viz_mask].reshape(-1, 3),
        colors=image_np[viz_mask].reshape(-1, 3)
    ))

    if not log_cameras:
        return

    h, w = image_np.shape[:2]
    rr.log(base_name, rr.Transform3D(translation=pose_np[:3, 3], mat3x3=pose_np[:3, :3]))
    rr.log(f"{base_name}/pinhole", rr.Pinhole(
        image_from_camera=intrinsics_np, height=h, width=w,
        camera_xyz=rr.ViewCoordinates.RDF, image_plane_distance=1.0
    ))
    rr.log(f"{base_name}/pinhole/rgb", rr.Image(image_np))
    if log_depth_mask:
        rr.log(f"{base_name}/pinhole/depth", rr.DepthImage(depth_z_np))
        rr.log(f"{base_name}/pinhole/mask", rr.SegmentationImage(viz_mask.astype(int)))


def transform_to_world_frame(preds, T_world_first, device):
    """Transform chunk predictions to world frame using first view's world pose."""
    for pred in preds:
        # Transform poses
        cam_quats = pred["cam_quats"]
        cam_trans = pred["cam_trans"]
        T_chunk = torch.eye(4, device=device).unsqueeze(0).repeat(cam_quats.shape[0], 1, 1)
        T_chunk[:, :3, :3] = quaternion_to_rotation_matrix(cam_quats)
        T_chunk[:, :3, 3] = cam_trans
        T_world = T_world_first @ T_chunk
        pred["cam_quats"] = rotation_matrix_to_quaternion(T_world[:, :3, :3])
        pred["cam_trans"] = T_world[:, :3, 3]

        # Transform pts3d
        pts3d = pred["pts3d"]  # (B, H, W, 3)
        B, H, W, _ = pts3d.shape
        pts_flat = pts3d.reshape(B, -1, 3)  # (B, H*W, 3)
        R = T_world_first[:3, :3]  # (3, 3)
        t = T_world_first[:3, 3]  # (3,)
        pts_world = (R @ pts_flat.transpose(1, 2)).transpose(1, 2) + t
        pred["pts3d"] = pts_world.reshape(B, H, W, 3)


def accumulate_sim3_transforms(transforms):
    """
    Accumulate adjacent SIM(3) transforms into transforms from frame 0 to frame k.

    Args:
        transforms: list of (s, R, t) tuples

    Returns:
        Cumulative transforms list, each (s_cum, R_cum, t_cum)
    """
    if not transforms:
        return []

    cumulative = [transforms[0]]

    for i in range(1, len(transforms)):
        s_prev, R_prev, t_prev = cumulative[i - 1]
        s_next, R_next, t_next = transforms[i]

        R_cum = R_prev @ R_next
        s_cum = s_prev * s_next
        t_cum = s_prev * (R_prev @ t_next) + t_prev

        cumulative.append((s_cum, R_cum, t_cum))

    return cumulative


def compute_sim3_from_overlap(stored_preds, preds, overlap_indices, device, use_conf=False):
    """
    Compute SIM(3) transform aligning new predictions to stored predictions.

    Args:
        stored_preds: List of stored predictions (target frame)
        preds: List of new predictions from current chunk (source frame)
        overlap_indices: List of (global_idx, local_idx) for overlap views
        device: torch device
        use_conf: If True, use confidence-weighted alignment (like DA3)

    Returns:
        (s, R, t) - SIM(3) parameters: scale, rotation (3x3), translation (3,)
        Returns identity transform if alignment fails
    """
    all_src_pts = []
    all_tgt_pts = []
    all_weights = []
    all_conf1 = []
    all_conf2 = []

    for global_idx, local_idx in overlap_indices:
        stored = stored_preds[global_idx]
        new = preds[local_idx]

        # Get masks - use non_ambiguous_mask as validity indicator
        stored_mask = stored["non_ambiguous_mask"][0].cpu().numpy()  # (H, W)
        new_mask = new["non_ambiguous_mask"][0].cpu().numpy()
        common_mask = stored_mask & new_mask

        if not common_mask.any():
            continue

        # Get confidence values if using confidence weighting
        if use_conf:
            stored_conf = stored["conf"][0].cpu().numpy()  # (H, W)
            new_conf = new.get("conf", new["non_ambiguous_mask"].float())[0].cpu().numpy()
            all_conf1.append(stored_conf[common_mask])
            all_conf2.append(new_conf[common_mask])

        # Extract corresponding points
        stored_pts = stored["pts3d"][0].cpu().numpy()  # (H, W, 3)
        new_pts = new["pts3d"][0].cpu().numpy()

        src_pts = new_pts[common_mask]      # Source: new chunk predictions
        tgt_pts = stored_pts[common_mask]   # Target: stored predictions

        all_src_pts.append(src_pts)
        all_tgt_pts.append(tgt_pts)

    if not all_src_pts:
        print("    Warning: No valid overlap points for SIM(3) alignment, using identity")
        return 1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    src = np.concatenate(all_src_pts, axis=0)
    tgt = np.concatenate(all_tgt_pts, axis=0)

    if use_conf and all_conf1:
        # Confidence-weighted alignment (DA3 style)
        conf1 = np.concatenate(all_conf1, axis=0)
        conf2 = np.concatenate(all_conf2, axis=0)

        # Dynamic threshold: min(median(conf1), median(conf2)) * 0.1
        conf_threshold = min(np.median(conf1), np.median(conf2)) * 0.1

        # Filter by confidence threshold
        conf_mask = (conf1 > conf_threshold) & (conf2 > conf_threshold)
        if not conf_mask.any():
            print("    Warning: No points above confidence threshold, using all points with uniform weights")
            weights = np.ones(src.shape[0], dtype=np.float32)
        else:
            src = src[conf_mask]
            tgt = tgt[conf_mask]
            conf1 = conf1[conf_mask]
            conf2 = conf2[conf_mask]
            # Combined confidence weight: sqrt(conf1 * conf2)
            weights = np.sqrt(conf1 * conf2).astype(np.float32)
            print(f"    Confidence threshold: {conf_threshold:.4f}, {conf_mask.sum()} / {len(conf_mask)} points above threshold")
    else:
        # Uniform weights (non_ambiguous_mask only)
        weights = np.ones(src.shape[0], dtype=np.float32)

    print(f"    SIM(3) alignment using {len(src)} corresponding points")

    # Use robust estimation with Huber loss
    s, R, t = robust_weighted_estimate_sim3_torch(
        src, tgt, weights,
        delta=0.1, max_iters=10, tol=1e-6, align_method="sim3"
    )

    return s, R, t


def apply_sim3_to_preds(preds, s, R, t, device):
    """
    Apply SIM(3) transform to chunk predictions (pts3d and camera poses).

    Args:
        preds: List of prediction dicts for the chunk
        s: Scale factor (numpy scalar)
        R: 3x3 rotation matrix (numpy)
        t: 3D translation vector (numpy)
        device: torch device
    """
    R_torch = torch.from_numpy(R).float().to(device)
    t_torch = torch.from_numpy(t).float().to(device)
    s_float = float(s)

    for pred in preds:
        # Transform pts3d: x' = s * R @ x + t
        pts3d = pred["pts3d"]  # (B, H, W, 3)
        B, H, W, _ = pts3d.shape
        pts_flat = pts3d.reshape(B, -1, 3)  # (B, H*W, 3)
        pts_transformed = s_float * (pts_flat @ R_torch.T) + t_torch
        pred["pts3d"] = pts_transformed.reshape(B, H, W, 3)

        # Transform camera poses
        cam_quats = pred["cam_quats"]  # (B, 4)
        cam_trans = pred["cam_trans"]  # (B, 3)

        # Convert to rotation matrices
        R_cam = quaternion_to_rotation_matrix(cam_quats)  # (B, 3, 3)

        # New rotation: R_new = R @ R_cam
        R_new = R_torch @ R_cam

        # New translation: t_new = s * R @ t_cam + t
        t_new = s_float * (cam_trans @ R_torch.T) + t_torch

        pred["cam_quats"] = rotation_matrix_to_quaternion(R_new)
        pred["cam_trans"] = t_new


@torch.inference_mode()
def run_chunk_inference(model, chunk_views, device, use_amp, amp_dtype):
    """Run model.forward() on a chunk of views."""
    # Move to device
    for view in chunk_views:
        for k, v in view.items():
            if isinstance(v, torch.Tensor):
                view[k] = v.to(device, non_blocking=True)

    # Configure model for inference with geometric inputs
    model._configure_geometric_input_config(
        use_calibration=True, use_depth=True, use_pose=True,
        use_depth_scale=True, use_pose_scale=True
    )

    with torch.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
        preds = model.forward(chunk_views, memory_efficient_inference=True, minibatch_size=1)

    model._restore_original_geometric_input_config()
    return preds


def get_parser():
    parser = argparse.ArgumentParser(description="MapAnything: Streaming Inference")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to images folder")
    parser.add_argument("--window_size", type=int, default=8, help="Images per chunk")
    parser.add_argument("--stride", type=int, default=4, help="New images per chunk")
    parser.add_argument("--num_iterations", type=int, default=1, help="Times to process all chunks (>1 for looped inference)")
    parser.add_argument("--apache", action="store_true", help="Use Apache 2.0 model")
    parser.add_argument("--viz", action="store_true", help="Enable Rerun visualization")
    parser.add_argument("--no_log_cameras", action="store_true", help="Disable camera logging (only log pointclouds)")
    parser.add_argument("--log_depth_mask", action="store_true", help="Log depth and mask for cameras (off by default)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--debug", action="store_true", help="Print debug stats comparing input vs output predictions")
    parser.add_argument("--no_prior_preds", action="store_true", help="Disable using prior predictions as input for next chunk")
    parser.add_argument("--pose_transform", action="store_true", help="Transform chunk poses to global world frame (chunk 0 view 0 as origin)")
    parser.add_argument("--sim3_align", action="store_true", help="Align consecutive chunks using SIM(3) from overlapping views")
    parser.add_argument("--conf_align", action="store_true", help="Use confidence-weighted alignment (requires --sim3_align)")
    return parser


def main():
    parser = get_parser()
    script_add_rerun_args(parser)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine amp dtype
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Load model
    model_name = "facebook/map-anything-apache" if args.apache else "facebook/map-anything"
    print(f"Loading model: {model_name}")
    model = MapAnything.from_pretrained(model_name).to(device)
    model.eval()

    # Load images
    print(f"Loading images from: {args.image_folder}")
    all_views = load_images(args.image_folder)
    num_images = len(all_views)
    print(f"Loaded {num_images} images")

    # Validate args
    if args.stride > args.window_size:
        args.stride = args.window_size
        warnings.warn(f"stride > window_size, setting stride = window_size = {args.window_size}")

    # Initialize Rerun
    if args.viz:
        rr.script_setup(args, "MapAnything_Streaming_Inference")
        rr.log("streaming", rr.ViewCoordinates.RDF, static=True)

    # Storage for predictions (indexed by global view index)
    # Each entry: {"ray_directions": ..., "depth_along_ray": ..., "cam_quats": ..., "cam_trans": ..., "non_ambiguous_mask": ..., "pts3d": ...}
    stored_preds = [None] * num_images

    # Storage for SIM(3) transforms per chunk (for --sim3_align)
    chunk_sim3_transforms = []

    for iteration in range(args.num_iterations):
        print(f"\n=== Iteration {iteration} ===")

        # Reset SIM(3) transforms for each iteration
        chunk_sim3_transforms.clear()

        # Generate chunk indices (stop once a chunk covers all remaining images)
        chunk_start_indices = []
        start = 0
        while start < num_images:
            chunk_start_indices.append(start)
            if start + args.window_size >= num_images:
                break  # This chunk covers to the end
            start += args.stride

        for chunk_idx, start_idx in enumerate(chunk_start_indices):
            end_idx = min(start_idx + args.window_size, num_images)
            chunk_global_indices = list(range(start_idx, end_idx))

            if args.verbose:
                print(f"  Chunk {chunk_idx}: views {chunk_global_indices}")

            # Build chunk views
            chunk_views = []
            for global_idx in chunk_global_indices:
                base_view = all_views[global_idx]
                view = {
                    "img": base_view["img"].clone(),
                    "data_norm_type": base_view["data_norm_type"],
                }

                # Add prior predictions if available (unless disabled)
                if not args.no_prior_preds and stored_preds[global_idx] is not None:
                    sp = stored_preds[global_idx]
                    # Mask depth to zero where non_ambiguous_mask is False
                    masked_depth = sp["depth_along_ray"].clone()
                    mask_expanded = sp["non_ambiguous_mask"].unsqueeze(-1)
                    masked_depth[~mask_expanded] = 0.0

                    view["ray_directions_cam"] = sp["ray_directions"]
                    view["depth_along_ray"] = masked_depth
                    view["camera_pose_quats"] = sp["cam_quats"]
                    view["camera_pose_trans"] = sp["cam_trans"]
                    view["is_metric_scale"] = torch.ones(1, dtype=torch.bool)

                chunk_views.append(view)

            # Count views with/without priors for logging
            num_with_priors = sum(1 for idx in chunk_global_indices if not args.no_prior_preds and stored_preds[idx] is not None)
            num_without_priors = len(chunk_global_indices) - num_with_priors
            print(f"    Prior injection: {num_with_priors} views with priors, {num_without_priors} views without priors")

            # Run inference
            preds = run_chunk_inference(model, chunk_views, device, use_amp=True, amp_dtype=amp_dtype)

            # Transform predictions to world frame if enabled
            if args.pose_transform:
                first_global_idx = chunk_global_indices[0]
                if chunk_idx > 0 and stored_preds[first_global_idx] is not None:
                    sp = stored_preds[first_global_idx]
                    T_world_first = torch.eye(4, device=device)
                    T_world_first[:3, :3] = quaternion_to_rotation_matrix(sp["cam_quats"])
                    T_world_first[:3, 3] = sp["cam_trans"].squeeze(0)
                    transform_to_world_frame(preds, T_world_first, device)
                    print(f"    Transformed chunk {chunk_idx} to world frame")

            # SIM(3) alignment: align chunk to chunk 0's coordinate frame
            if args.sim3_align:
                if chunk_idx == 0:
                    # Identity transform for first chunk
                    chunk_sim3_transforms.append((1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)))
                else:
                    # Compute SIM(3) from overlapping views
                    overlap_count = args.window_size - args.stride
                    overlap_indices = [
                        (chunk_global_indices[i], i)
                        for i in range(min(overlap_count, len(chunk_global_indices)))
                        if stored_preds[chunk_global_indices[i]] is not None
                    ]

                    if overlap_indices:
                        # Compute SIM(3) from new preds to stored preds
                        s, R, t = compute_sim3_from_overlap(stored_preds, preds, overlap_indices, device, use_conf=args.conf_align)
                        chunk_sim3_transforms.append((s, R, t))

                        # Accumulate transforms and apply cumulative to this chunk
                        cumulative = accumulate_sim3_transforms(chunk_sim3_transforms)
                        s_cum, R_cum, t_cum = cumulative[-1]
                        apply_sim3_to_preds(preds, s_cum, R_cum, t_cum, device)

                        if args.verbose:
                            print(f"    Applied SIM(3): s={s_cum:.4f}, |t|={np.linalg.norm(t_cum):.4f}")
                    else:
                        # No overlap available, use identity
                        print("    Warning: No overlap views available for SIM(3), using identity")
                        chunk_sim3_transforms.append((1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)))

            # Store predictions and log to Rerun
            overlap_count = args.window_size - args.stride
            for i, (global_idx, pred) in enumerate(zip(chunk_global_indices, preds)):
                # Determine if this is a new view or an overlap view
                is_new_view = (chunk_idx == 0) or (i >= overlap_count)

                # Debug: compare with previous prediction if available (only for new views)
                if args.debug and is_new_view and stored_preds[global_idx] is not None:
                    compute_debug_stats(stored_preds[global_idx], pred, global_idx, iteration)

                # Only store predictions for NEW views, not overlap views
                # Overlap views should keep their original prediction from when they were first processed
                if is_new_view:
                    stored_preds[global_idx] = {
                        "ray_directions": pred["ray_directions"].detach(),
                        "depth_along_ray": pred["depth_along_ray"].detach(),
                        "cam_quats": pred["cam_quats"].detach(),
                        "cam_trans": pred["cam_trans"].detach(),
                        "non_ambiguous_mask": pred["non_ambiguous_mask"].detach(),
                        "pts3d": pred["pts3d"].detach(),
                        "conf": pred.get("conf", pred["non_ambiguous_mask"].float()).detach(),
                    }

                # Log to Rerun
                if args.viz:
                    if is_new_view or iteration > 0:
                        # Get numpy arrays for visualization
                        img_torch = chunk_views[i]["img"]
                        img_np = rgb(img_torch, chunk_views[i]["data_norm_type"][0])[0]  # Remove batch dim

                        pts3d_np = pred["pts3d"][0].cpu().numpy()
                        depth_z_np = pred["pts3d_cam"][0, :, :, 2].cpu().numpy()
                        non_amb_mask_np = pred["non_ambiguous_mask"][0].cpu().numpy()

                        # Compute viz mask with edge masking
                        viz_mask = compute_viz_mask(pts3d_np, depth_z_np, non_amb_mask_np)

                        # Get pose and intrinsics
                        cam_quats = pred["cam_quats"]
                        cam_trans = pred["cam_trans"]
                        pose = torch.eye(4)
                        pose[:3, :3] = quaternion_to_rotation_matrix(cam_quats)[0].cpu()
                        pose[:3, 3] = cam_trans[0].cpu()
                        intrinsics = recover_pinhole_intrinsics_from_ray_directions(pred["ray_directions"])[0].cpu().numpy()

                        rr.set_time("iteration", sequence=iteration)
                        base_name = f"streaming/iter_{iteration}/view_{global_idx}"
                        pts_name = f"streaming/iter_{iteration}/pointcloud_{global_idx}"
                        log_to_rerun(img_np, depth_z_np, pose.numpy(), intrinsics, pts3d_np, viz_mask,
                                     base_name, pts_name, log_cameras=not args.no_log_cameras,
                                     log_depth_mask=args.log_depth_mask)

        print(f"Iteration {iteration} complete")

    print("\nStreaming inference complete!")
    if args.viz:
        print("Check Rerun viewer for results.")


if __name__ == "__main__":
    main()