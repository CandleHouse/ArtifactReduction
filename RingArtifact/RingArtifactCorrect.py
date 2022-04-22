'''
    This example uses crip to perform ring artifact correction on projections
    using existing projection and FDK reconstruction algorithm.

    by CandleHouse @ https://github.com/CandleHouse/ArtifactReduction
'''

from crip.io import *
from crip.preprocess import *
from tqdm import tqdm
from numba import jit
import tifffile


def airRaw2tif(raw_path, tif_path, shape: tuple, offset=0, filter=False):
    """
        smooth to reduce ring artifact
    """
    @jit(nopython=True)
    def MedianFilter(img, k=3):
        height, width = img.shape
        edge = k // 2
        new_arr = np.zeros((height, width), dtype=np.float32)
        for i in range(height):
            for j in range(width):
                if i <= edge - 1 or i >= height - 1 - edge or j <= edge - 1 or j >= width - edge - 1:
                    new_arr[i, j] = img[i, j]
                else:
                    new_arr[i, j] = np.median(img[i - edge:i + edge + 1, j - edge:j + edge + 1])
        return new_arr

    H, W = shape
    for file, name in tqdm(listDirectory(raw_path, style='both')):
        proj = np.fromfile(file, dtype=np.uint16)[offset:].reshape(H, W)
        # smooth with median filter
        if filter:
            k = 5
            nPadU, nPadD, nPadL, nPadR = k // 2, k // 2, k // 2, k // 2
            temp = np.pad(proj.astype(np.float32), ((nPadU, nPadD), (nPadL, nPadR)), mode='reflect')
            proj = MedianFilter(temp, k)
            proj = proj[nPadU:nPadU+H, nPadL:nPadL+W]

        tifffile.imwrite(tif_path + name.replace('.raw', '.tif'), proj)


if __name__ == '__main__':
    raw_path = r''
    tif_path = r''
    H, W = 1944, 3072

    # 1. air smooth
    airRaw2tif(raw_path=raw_path,
               tif_path=tif_path,
               shape=(H, W), offset=256, filter=True)

    # 2. image reconstruction
    # Use your own recon tool.
