import cv2
import matplotlib.pyplot as plt
import numpy as np

# from https://www.kaggle.com/robertkag/rle-to-mask-converter
def rle_to_mask(rle_string, height, width):
    '''
    convert RLE(run length encoding) string to numpy array

    Parameters: 
    rle_string (str): string of rle encoded mask
    height (int): height of the mask
    width (int): width of the mask 

    Returns: 
    numpy.array: numpy array of the mask
    '''
    
    rows, cols = height, width
    
    if rle_string == -1:
        return np.zeros((height, width))
    else:
        rle_numbers = [int(num_string) for num_string in rle_string.split(' ')]
        rle_pairs = np.array(rle_numbers).reshape(-1,2)
        img = np.zeros(rows*cols, dtype=np.uint8)
        for index, length in rle_pairs:
            index -= 1
            img[index:index+length] = 255
        img = img.reshape(cols,rows)
        img = img.T
        return img



# path_to_img = r'severstal-steel-defect-detection\train_images\0a1cade03.jpg'
# img = cv2.imread(path_to_img)
import pandas as pd
train_df = pd.read_csv('severstal-steel-defect-detection/train.csv')
train_df['img_id'] = train_df.ImageId_ClassId.str.split("_", n = 1, expand = True)[0]
train_df['class_id'] = train_df.ImageId_ClassId.str.split("_", n = 1, expand = True)[1]
train_df= train_df.drop(['ImageId_ClassId'],axis=1)
train_df = train_df.dropna()

