import numpy as np
from scipy.interpolate import NearestNDInterpolator  

array = np.array(([100,2,0,0],
                  [2,3,0,0],
                  [0,5,7,8]), dtype=np.int64)

print("Before inpainting:")
print(array)

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