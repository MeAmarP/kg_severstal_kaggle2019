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

def displayImageData(in_df,no_of_sample=9):
    """Disaplay Sample Image data
    
    Arguments:
        in_df {pandas dataframe} -- Input train Dataframe with Img_Path Column
    
    Keyword Arguments:
        no_of_sample {int} -- Number of Image samples to display (default: {9})
    """
    fig = plt.figure("EDA Image Data")
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    fig.suptitle('Understanding Image Data', fontsize=16)

    
    for n, path in enumerate(in_df.img_id[:no_of_sample]):
        label = str(in_df.class_id[n])
        ImgTitle = in_df.img_id[n] +'-'+label
        Img = cv2.imread('severstal-steel-defect-detection/train_images/'+path)
        #Note: Adjust the subplot grid so nb_of_images fits pefect ina grid.
        ax = fig.add_subplot(3,3,(n+1))
        plt.imshow(cv2.cvtColor(Img, cv2.COLOR_BGR2RGB))
        ax.set_title(ImgTitle)
        plt.axis('off')
    plt.show()
    return


# path_to_img = r'severstal-steel-defect-detection\train_images\0a1cade03.jpg'
# img = cv2.imread(path_to_img)
import pandas as pd
train_df = pd.read_csv('severstal-steel-defect-detection/train.csv')
train_df['img_id'] = train_df.ImageId_ClassId.str.split("_", n = 1, expand = True)[0]
train_df['class_id'] = train_df.ImageId_ClassId.str.split("_", n = 1, expand = True)[1]
train_df= train_df.drop(['ImageId_ClassId'],axis=1)
train_df = train_df.dropna()
train_df = train_df.reset_index(drop=True)
print(train_df.head())

# r'severstal-steel-defect-detection\train_images\0a1cade03.jpg'
displayImageData(train_df)
