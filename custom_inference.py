import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from inference import Inferencer
import matplotlib.pyplot as plt

import pyrealsense2 as rs

def draw_point_cloud(rgb_np, depth_np, cam_intrinsics, complete_pcd=False):
    h, w, _ = rgb_np.shape
    # print(h,w)
    if(complete_pcd==True):
        inferencer = Inferencer()
        depth_np = depth_np / 1000
        depth_np, depth_ori = inferencer.inference(rgb_np, depth_np, depth_coefficient=3, inpainting=True)
        # depth_np *= 1000 
        print(f"Sample inference depth: {depth_np}")
    img = o3d.geometry.Image(rgb_np.astype(np.uint8))
    depth = o3d.geometry.Image(depth_np)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(img, depth, depth_trunc=3000.0, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd, o3d.camera.PinholeCameraIntrinsic(w,h,cam_intrinsics.fx, cam_intrinsics.fy, cam_intrinsics.ppx, cam_intrinsics.ppy))
    # plt.subplot(1, 2, 1)
    # plt.title('grayscale image')
    # plt.imshow(rgbd.color)
    # plt.subplot(1, 2, 2)
    # plt.title('depth image')
    # plt.imshow(rgbd.depth)
    # plt.show()
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([pcd], window_name='Point Cloud')
    # o3d.io.write_point_cloud("data/d435i/10_239722074298_transcg.pcd", pcd)

def get_intrinsics():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    print(device)

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    # Get stream profile and camera intrinsics
    profile = pipeline.get_active_profile()
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    depth_intrinsics = depth_profile.get_intrinsics()

    return depth_intrinsics

def main():
    cam_intrinsics = get_intrinsics()
    print(f'cam_intrinsics: {cam_intrinsics}')

    rgb_np = np.array(Image.open('data/d435i/10_239722074298.png'), dtype = np.float32)
    depth_np = np.load('data/d435i/10_239722074298.npy')
    print(f"Depth shape: {depth_np.shape}, Original depth: {depth_np}")

    # draw_point_cloud(rgb_np, depth_np, cam_intrinsics)
    draw_point_cloud(rgb_np, depth_np, cam_intrinsics, complete_pcd=True)

if __name__ == '__main__':
    main()