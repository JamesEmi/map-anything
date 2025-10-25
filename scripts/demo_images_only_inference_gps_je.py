# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
MapAnything Demo: Images-Only Inference with Visualization

Usage:
    python demo_images_only_inference.py --help
"""

import argparse
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import cv2
import rerun as rr
import torch
import trimesh
# import open3d

from mapanything.models import MapAnything
from mapanything.utils.geometry import depthmap_to_world_frame
from mapanything.utils.image import load_images
from mapanything.utils.viz import (
    predictions_to_glb,
    script_add_rerun_args,
)
# from mapanything.utils.gps_helpers_pymap import attach_translation_poses_from_gps
from gps_helpers_pymap import attach_translation_poses_from_gps


def log_data_to_rerun(
    image, depthmap, pose, intrinsics, pts3d, mask, base_name, pts_name, viz_mask=None
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
    rr.log(
        f"{base_name}/pinhole/depth",
        rr.DepthImage(depthmap),
    )
    if viz_mask is not None:
        rr.log(
            f"{base_name}/pinhole/mask",
            rr.SegmentationImage(viz_mask.astype(int)),
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
        "--memory_efficient_inference",
        action="store_true",
        default=False,
        help="Use memory efficient inference for reconstruction (trades off speed)",
    )
    parser.add_argument(
        "--apache",
        action="store_true",
        help="Use Apache 2.0 licensed model (facebook/map-anything-apache)",
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
    # Optional depth inputs (paired by filename stem)
    parser.add_argument(
        "--depth_folder",
        type=str,
        default=None,
        help="Optional folder containing depth maps (.npy meters or 16-bit .png millimeters)",
    )

    parser.add_argument(
        "--gps_csv",
        type=str,
        required=True,
        help="CSV with columns timestamp_ns,timestamp_s,latitude,longitude,altitude",
    )
    parser.add_argument(
        "--tolerance_ms",
        type=float,
        default=100.0,
        help="Max time difference to match image to GPS (ms)",
    )

    parser.add_argument("--save_ply", action="store_true", default=False, help="Save point cloud as PLY")
    parser.add_argument("--ply_output_path", type=str, default="output.ply", help="PLY output path")

    return parser


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

    # Initialize model from HuggingFace
    if args.apache:
        model_name = "facebook/map-anything-apache"
        print("Loading Apache 2.0 licensed MapAnything model...")
    else:
        model_name = "facebook/map-anything"
        print("Loading CC-BY-NC 4.0 licensed MapAnything model...")
    model = MapAnything.from_pretrained(model_name).to(device)

    # Load images
    print(f"Loading images from: {args.image_folder}")
    views = load_images(args.image_folder)
    print(f"Loaded {len(views)} views")

    supported_images_extensions = (".jpg", ".jpeg", ".png")
    image_paths = [
        os.path.join(args.image_folder, f)
        for f in sorted(os.listdir(args.image_folder))
        if f.lower().endswith(supported_images_extensions)
    ]
    # Attach translation-only poses from GPS
    matched, total = attach_translation_poses_from_gps(
        views=views,
        image_paths=image_paths,
        gps_csv_path=args.gps_csv,
        tolerance_ms=args.tolerance_ms,
    )
    print(f"Attached GPS translation poses to {matched}/{total} views (others defaulted to origin relative).")

    # Run model inference
    print("Running inference...")
    outputs = model.infer(
        views, memory_efficient_inference=args.memory_efficient_inference,
        ignore_calibration_inputs=False,
        ignore_depth_inputs=True,
        ignore_pose_inputs=False,
        ignore_depth_scale_inputs=True,
        ignore_pose_scale_inputs=False,
    )
    print("Inference complete!")

    # Prepare lists for GLB export if needed
    world_points_list = []
    images_list = []
    masks_list = []

    # Initialize Rerun if visualization is enabled
    if args.viz:
        print("Starting visualization...")
        viz_string = "MapAnything_Visualization"
        rr.script_setup(args, viz_string)
        rr.set_time("stable_time", sequence=0)
        rr.log("mapanything", rr.ViewCoordinates.RDF, static=True)

    # Loop through the outputs
    for view_idx, pred in enumerate(outputs):
        # Extract data from predictions
        depthmap_torch = pred["depth_z"][0].squeeze(-1)  # (H, W)
        intrinsics_torch = pred["intrinsics"][0]  # (3, 3)
        camera_pose_torch = pred["camera_poses"][0]  # (4, 4)

        # Compute new pts3d using depth, intrinsics, and camera pose
        pts3d_computed, valid_mask = depthmap_to_world_frame(
            depthmap_torch, intrinsics_torch, camera_pose_torch
        )

        # Convert to numpy arrays
        mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
        mask = mask & valid_mask.cpu().numpy()  # Combine with valid depth mask
        pts3d_np = pts3d_computed.cpu().numpy()
        image_np = pred["img_no_norm"][0].cpu().numpy()

        # Store data for GLB export if needed
        if args.save_glb or args.save_ply:
            world_points_list.append(pts3d_np)
            images_list.append(image_np)
            masks_list.append(mask)

        # Log to Rerun if visualization is enabled
        if args.viz:
            log_data_to_rerun(
                image=image_np,
                depthmap=depthmap_torch.cpu().numpy(),
                pose=camera_pose_torch.cpu().numpy(),
                intrinsics=intrinsics_torch.cpu().numpy(),
                pts3d=pts3d_np,
                mask=mask,
                base_name=f"mapanything/view_{view_idx}",
                pts_name=f"mapanything/pointcloud_view_{view_idx}",
                viz_mask=mask,
            )

    if args.viz:
        print("Visualization complete! Check the Rerun viewer.")


    if args.save_glb or args.save_ply:
        # Stack all views
        world_points = np.stack(world_points_list, axis=0)
        images = np.stack(images_list, axis=0)
        final_masks = np.stack(masks_list, axis=0)
        
    # Export GLB if requested
    if args.save_glb:
        print(f"Saving GLB file to: {args.output_path}")
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
        print("Skipping GLB export (--save_glb not specified)")

    if args.save_ply:
        vertices = world_points.reshape(-1, 3)
        colors = (images.reshape(-1, 3) * 255).astype(np.uint8)
        mask = final_masks.reshape(-1).astype(bool)
        vertices = vertices[mask]
        colors = colors[mask]

        # optional downsample to keep file small
        # max_points = 5_000_000
        # if vertices.shape[0] > max_points:
        #     sel = np.random.choice(vertices.shape[0], size=max_points, replace=False)
        #     vertices, colors = vertices[sel], colors[sel]

        trimesh.PointCloud(vertices=vertices, colors=colors).export(args.ply_output_path)
        print(f"Saved PLY to: {args.ply_output_path}")
    else:
        print("Skipping PLY export (--save_ply not specified)")

if __name__ == "__main__":
    main()
