import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# %%
"""
Initial decideM function that takes an image and decides if it's a T or an M
"""
def decideM(img:Image, imgName):
    """
        returns 0 if the image is labeled with T, returns 1 if image is M
    """
    h = img.height
    cropped = np.asanyarray(img.crop((300, h-80, 370, h-15)))
    nrBlackPixels = np.count_nonzero(cropped < 50)
   
    if (nrBlackPixels != 3180 and nrBlackPixels != 4200 and nrBlackPixels != 4197):
        plt.imshow(img)
        plt.imshow
        plt.imshow(cropped)
        plt.show()
        print(f'Nr of black pixels is: {nrBlackPixels} \nFilename is: {imgName}')
    return nrBlackPixels > 3690


# %%
"""
    loads all the images without a dataloader: takes about 24h for all the heron-images
"""
imagePropsList = []
files = os.listdir(DATA_DIR)
folders = [f for f in files if os.path.isdir(DATA_DIR+'/'+f)]

for folderName in tqdm(folders):
    folderPath = os.path.abspath(os.path.join(DATA_DIR, folderName))
    for imgName in os.listdir(folderPath):
        img = Image.open(os.path.abspath(os.path.join(folderPath, imgName)))
        imagePropsList.append([imgName, folderName, decideM(img, imgName)])  