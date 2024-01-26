import numpy as np
import cv2
import open3d as o3d
from PIL import Image
import matplotlib.pyplot as plt
from scipy.interpolate import NearestNDInterpolator  

# array = np.array(([100,2,0,0],
#                   [2,3,0,0],
#                   [0,5,7,8]), dtype=np.int64)

def nearest_interpolation(array):
    mask = np.where(array > 0)
    print(mask)

    if mask[0].shape[0] != 0:
        mask_T = np.transpose(mask)
        print(mask_T)
        print(array[mask])
        interp = NearestNDInterpolator(np.transpose(mask), array[mask])
        depth = interp(*np.indices(array.shape))
        
    print("After inpainting:")
    print(depth)

def bilinear_interpolation(img, mask):
    # Open the image.
    img = cv2.imread('data/d435i/10_239722074298.png')
    
    # Load the mask.
    mask = cv2.imread('data/d435i/mask_10_239722074298.png', 0)
    
    # Inpaint.
    dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
    
    # Write the output.
    cv2.imwrite('inpainted.png', dst)

def show_depth_image(path):
    image = Image.open('data/d435i/mask_10_239722074298.png').convert('L')
    plt.imshow(image, cmap='gray')
    plt.show()    

if __name__ == '__main__':
    # array = np.array([[3, 0, 0], 
    #               [0, 4, 0], 
    #               [5, 6, 0]])

    # print("Before inpainting:")
    # print(array)

    # rgb_np = np.array(Image.open('data/d435i/10_239722074298.png'), dtype = np.float32)
    # depth_np = np.load('data/d435i/10_239722074298.npy')
    # print(f"Original depth: {depth_np.shape}")

    # for i in range(depth_np.shape[0]):
    #     for j in range(depth_np.shape[1]):
    #         if depth_np[i, j].sum() > 0:
    #             depth_np[i, j] = 0
    #         else:
    #             depth_np[i, j] = 255
    # print(depth_np)

    # mask = depth_np.copy()
    # # mask = np.where(depth_np>0.0, 1.0, depth_np)
    # print(f"Mask depth: {mask}")
    # depth_map = np.array(mask, dtype=np.float32)

    # h, w, _ = rgb_np.shape
    # # print(h,w)
    # img = o3d.geometry.Image(rgb_np.astype(np.uint8))
    # depth = o3d.geometry.Image(depth_map)
    # rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(img, depth, depth_trunc=3000.0, convert_rgb_to_intensity=False)
    # plt.subplot(1, 2, 1)
    # plt.title('grayscale image')
    # plt.imshow(rgbd.color)
    # plt.subplot(1, 2, 2)
    # plt.title('depth image')
    # plt.imshow(rgbd.depth)
    # plt.show()

    # cv2.imwrite('data/d435i/mask_10_239722074298.png', mask)

    # bilinear_interpolation(depth_np, mask)
    # nearest_interpolation(array)

    # show_depth_image('data/d435i/mask_10_239722074298.png')

    image = Image.open('data/d435i/mask_10_239722074298.png').convert('L')
    plt.imshow(image, cmap='gray')
    plt.show()