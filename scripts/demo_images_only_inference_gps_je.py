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
import pymap3d as pm

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
from gps_helpers_pymap import read_gps_csv, match_images_to_gps, attach_translation_poses_from_gps
from mapanything.utils.geometry import get_coord_system_transform




def _downsample_points_and_colors(points: np.ndarray, colors: np.ndarray, max_points: int, rng_seed: int = 0):
    """Randomly downsample positions and matching colors to at most max_points."""
    if max_points is None or max_points <= 0 or points.shape[0] <= max_points:
        return points, colors
    rng = np.random.default_rng(rng_seed)
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx], colors[idx]


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
        default=20.0,
        help="Max time difference to match image to GPS (ms, default: 20)",
    )

    parser.add_argument("--save_ply", action="store_true", default=False, help="Save point cloud as PLY")
    parser.add_argument("--ply_output_path", type=str, default="output.ply", help="PLY output path")
    parser.add_argument(
        "--max_points",
        type=int,
        default=2_000_000,
        help="Maximum number of points for final combined point cloud (for viz/PLY)",
    )

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
    original_total = len(views)
    matched, total = attach_translation_poses_from_gps(
        views=views,
        image_paths=image_paths,
        gps_csv_path=args.gps_csv,
        tolerance_ms=args.tolerance_ms,
    )
    post_total = len(views)
    dropped = max(0, original_total - post_total)
    unmatched_rgb_only = max(0, post_total - matched)
    print(
        f"Attached GPS translation poses to {matched}/{post_total} views; "
        f"dropped {dropped} pre-match views; {unmatched_rgb_only} RGB-only views."
    )

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
        rr.set_time("build", sequence=0)
        rr.log("mapanything", rr.ViewCoordinates.RDF, static=True)

    # GPS debug viz
    gps_rows = read_gps_csv(args.gps_csv)
    origin = gps_rows[0]
    matches = match_images_to_gps(
        image_paths=image_paths,
        gps=gps_rows,
        tolerance_ns=int(args.tolerance_ms * 1e6)
    )
    # gather matched GPS traj for the kept views
    gps_rdf = []
    gps_enu = []
    # Accumulators for viewer & final outputs
    all_pts = []
    all_cols = []
    est_traj = []
    # Loop through the outputs
    for view_idx, pred in enumerate(outputs):
        # Extract data from predictions
        depthmap_torch = pred["depth_z"][0].squeeze(-1)  # (H, W)
        intrinsics_torch = pred["intrinsics"][0]  # (3, 3)
        camera_pose_torch = pred["camera_poses"][0]  # (4, 4)

        R = camera_pose_torch[:3, :3].cpu().numpy()
        t = camera_pose_torch[:3, 3].cpu().numpy()
        axis_scale = 0.5  # tune
        # triad = np.stack([t, t + R[:,0]*axis_scale, t, t + R[:,1]*axis_scale, t, t + R[:,2]*axis_scale]).reshape(3,2,3)
        xw, yw, zw = R[:, 0] * axis_scale, R[:, 1] * axis_scale, R[:, 2] * axis_scale
        origins = np.stack([t, t, t], axis=0)
        vectors = np.stack([xw, yw, zw], axis=0)
        colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
        rr.log("mapanything/camera/axes", rr.Arrows3D(origins=origins, vectors=vectors, colors=colors))
        rr.log("diag/camera_axes",
            rr.Arrows3D(origins=np.array([t,t,t]), vectors=R.T*axis_scale,
                        colors=np.array([[255,0,0],[0,255,0],[0,0,255]], dtype=np.uint8)))               


        # Compute new pts3d using depth, intrinsics, and camera pose
        pts3d_computed, valid_mask = depthmap_to_world_frame(
            depthmap_torch, intrinsics_torch, camera_pose_torch
        )

        # Convert to numpy arrays
        mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
        mask = mask & valid_mask.cpu().numpy()  # Combine with valid depth mask
        pts3d_np = pts3d_computed.cpu().numpy()
        image_np = pred["img_no_norm"][0].cpu().numpy()

        # Store data for GLB export only (viewer/PLY use accumulators below)
        if args.save_glb:
            world_points_list.append(pts3d_np)
            images_list.append(image_np)
            masks_list.append(mask)

        # Log to Rerun if visualization is enabled
        if args.viz:
            # advance time on the 'build' timeline
            rr.set_time("build", sequence=view_idx)

            # incrementally log estimated trajectory (upto current step)
            current_t = camera_pose_torch[:3, 3].cpu().numpy()
            est_traj.append(current_t)
            est_arr = np.asarray(est_traj, dtype=np.float32)
            rr.log(
                "est/traj_points",
                rr.Points3D(positions=est_arr, radii=0.03, colors=np.array([[0, 255, 0]], dtype=np.uint8)),
            )
            rr.log(
                "est/traj_line",
                rr.LineStrips3D(strips=[est_arr]),
            )

            view = views[view_idx]
            idx_v = int(view.get("idx", 0))
            g = matches.get(image_paths[idx_v])
            if g is not None:
                e, n, u = pm.geodetic2enu(g.lat, g.lon, g.alt, origin.lat, origin.lon, origin.alt)
                # T_enu2rdf = get_coord_system_transform("RFU", "RDF")
                trans_enu = torch.tensor([float(e), float(n), float(u)], dtype=torch.float32)

                # 3D (about U axis): rotates [E, N, U] -> [N, -E, U]
                # my take [E, N, U] -> [-N, E, U]
                Rz_cw90 = torch.tensor([
                    [0.0,  1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0,  0.0, 1.0],
                ], dtype=torch.float32)
                trans_enu_rot = Rz_cw90 @ trans_enu
                # trans_rdf = T_enu2rdf @ trans_enu
                gps_enu.append(trans_enu_rot.cpu().numpy())
                # gps_rdf.append(trans_rdf.cpu().numpy())
                gps_arr = np.asarray(gps_enu, dtype=np.float32)
                # rr.log("gps/enu2rdf_points", rr.Points3D(positions=gps_arr))
                # rr.log("gps/enu2rdf_line", rr.LineStrips3D(strips=[gps_arr]))
                rr.log("gps/enu_points", rr.Points3D(positions=gps_arr))
                rr.log("gps/enu_line", rr.LineStrips3D(strips=[gps_arr]))

            # accumulate for 'built-out' playback
            pts_now = pts3d_np[mask].reshape(-1, 3)
            cols_now = (image_np[mask].reshape(-1, 3) * 255).astype("uint8")
            all_pts.append(pts_now)
            all_cols.append(cols_now)

            # log stable camera and pinhole
            rr.log("mapanything/camera/pinhole/rgb", rr.Image(image_np))
            rr.log(
                "mapanything/camera",
                rr.Transform3D(
                    translation=camera_pose_torch[:3, 3].cpu().numpy(),
                    mat3x3=camera_pose_torch[:3, :3].cpu().numpy(),
                ),
            )
            rr.log(
                "mapanything/camera/pinhole",
                rr.Pinhole(
                    image_from_camera=intrinsics_torch.cpu().numpy(),
                    height=image_np.shape[0],
                    width=image_np.shape[1],
                    camera_xyz=rr.ViewCoordinates.RDF,
                ),
            )
            # per-step points
            rr.log("mapanything/pointcloud_step", rr.Points3D(positions=pts_now, colors=cols_now))

    if args.viz:
        print("Visualization complete! Check the Rerun viewer.")
        # one final full map (downsample to keep size in check)
        final_pts = np.concatenate(all_pts, axis=0) if len(all_pts) > 0 else np.empty((0, 3))
        final_cols = np.concatenate(all_cols, axis=0) if len(all_cols) > 0 else np.empty((0, 3), dtype=np.uint8)

        # run once over concatenated points all_pts (Nx3)
        P = final_pts - final_pts.mean(0, keepdims=True)
        _, _, Vt = np.linalg.svd(P, full_matrices=False)
        normal = Vt[-1]  # unit normal
        print("Plane normal ~", normal, "angles to axes:",
            "X", np.degrees(np.arccos(np.clip(np.abs(normal[0]), -1, 1))),
            "Y", np.degrees(np.arccos(np.clip(np.abs(normal[1]), -1, 1))),
            "Z", np.degrees(np.arccos(np.clip(np.abs(normal[2]), -1, 1))))

        # optional subsample
        max_points = args.max_points if hasattr(args, "max_points") else 2_000_000
        if final_pts.shape[0] > max_points:
            sel = np.random.choice(final_pts.shape[0], size=max_points, replace=False)
            final_pts = final_pts[sel]
            final_cols = final_cols[sel]

        rr.set_time("build", sequence=len(outputs))
        rr.log("mapanything/pointcloud_final", rr.Points3D(positions=final_pts, colors=final_cols))

    # (Optional) Saving of .rrd is handled by rerun script args externally if desired

    # Export GLB if requested
    if args.save_glb:
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
        print("Skipping GLB export (--save_glb not specified)")

    if args.save_ply:
        # use accumulated points/colors for PLY
        vertices = np.concatenate(all_pts, axis=0) if len(all_pts) > 0 else np.empty((0, 3))
        colors = np.concatenate(all_cols, axis=0) if len(all_cols) > 0 else np.empty((0, 3), dtype=np.uint8)
        # Downsample final cloud to max_points if needed
        max_points = args.max_points if hasattr(args, "max_points") else 2_000_000
        if vertices.shape[0] > max_points:
            sel = np.random.choice(vertices.shape[0], size=max_points, replace=False)
            vertices = vertices[sel]
            colors = colors[sel]

        trimesh.PointCloud(vertices=vertices, colors=colors).export(args.ply_output_path)
        print(f"Saved PLY to: {args.ply_output_path}")
    else:
        print("Skipping PLY export (--save_ply not specified)")

if __name__ == "__main__":
    main()