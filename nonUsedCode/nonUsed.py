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

# %%
"""
   find out how to decide if black-white?
"""
for path in ["/data/shared/herons/TinaDubach_data/GBU2/2017_GBU2_01310001.JPG", "/data/shared/herons/TinaDubach_data/GBU2/2017_GBU2_01310047.JPG"]: # first is rgb, next is grayscale
    print(f'Image: {path}')
    img = Image.open(path)
    img = np.asarray(img)
    print(img.shape)
    w, h, streams = img.shape
    meanStd = 0
    for i, j in zip(random.choices(range(w), k=10), random.choices(range(h-200), k=10)):
        pixelSum = 0
        randpix = np.array(img[i, j, :])
        print(randpix)
        std = randpix.std()
        print(std)
        meanStd += std 
        # for i, s in enumerate(range(streams)):
        #     pixelSum += img[i, j, s]
        #     print(f'stream {i} contains pixel val {img[i, j, s]}')
    print(meanStd)

# %%
"""
decides if the image is grayscale - slow
"""
# def isGrayscale(img: torch.Tensor): #img: 3 x h x w
#     print(img.shape)
#     _, h, w = img.shape
#     sumStd = 0
#     for i, j in zip(random.choices(range(h-200), k=10), random.choices(range(w), k=10)):
#         randpix = np.array(img[:, i, j])
#         print(randpix)
#         std = randpix.std()
#         print(std)
#         sumStd += std 
#     if sumStd < 20:
#         return True
#     return False

# %%
"""
Analize the data for interesting cameras
"""
cameraDataDF = pd.read_csv("/data/shared/herons/TinaDubach_data/CameraData_2017_july.csv", encoding='unicode_escape', on_bad_lines="warn", sep=";")
cameraDataDF.describe()
# %%
# cameraDataDF.head(10)
cameraDataDF.groupby(["camera"]).size()
cameraDataDF[cameraDataDF["fotocode"] == "2017_SBU4_01270033"]

