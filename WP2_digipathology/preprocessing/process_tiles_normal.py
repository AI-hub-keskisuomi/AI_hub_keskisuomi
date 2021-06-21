import numpy as np
import openslide
import cv2
import os
import PIL
from PIL import Image, ImageDraw, ImageFont

#==================================================================================
# process_tiles() runs the whole tiling script and saves the tiles to directory
#==================================================================================

# reads WSI-files, converts them to numpy array, tiles them and saves to directory
# input: source directory, save directory level, tile_size

def process_tiles(source_dir, save_dir, level, tile_size):
  
    for filename in os.listdir(source_dir):
        if filename.endswith(".tif"):
            img = openslide.OpenSlide(os.path.join(source_dir, filename))  
            img = np.array(img.read_region((0, 0), level, img.level_dimensions[level]))
            print(img.shape)
            tiles = tile_im(img, tile_size, filename, save_dir)[1]
        else:
            continue
    return tiles



# masks and tiles the masked image (array) to smaller tiles according to step size
# input: image array, step size, filename, save directory
# output: array of tile cordinates, saved arrays .npy-format

def tile_im(slide, step, filename, save_dir):
    
    m,n = slide.shape[0], slide.shape[1]

    mask = cv2.cvtColor(slide, cv2.COLOR_RGBA2RGB)
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2HSV)
    mask = mask[:, :, 1]
    
    # make mask
    _, tissue_mask = cv2.threshold(mask, 200, 250, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    tissue_mask = np.array(tissue_mask)

    mask = tissue_mask > 0

    mask0,mask1 = mask.any(0),mask.any(1)
    x_start,x_end = mask0.argmax(),n-mask0[::1].argmax()
    y_start,y_end = mask1.argmax(),m-mask1[::1].argmax()
    
    # adjusting the image-size to match step-size
    rem_x = (x_end - x_start)% step
    rem_y = (y_end - y_start)% step
    
    if (rem_x != 0):
        x_start = x_start + rem_x
    if (rem_y != 0):
        y_start = y_start + rem_y
        
    img_cropped = tissue_mask[y_start:y_end,x_start:x_end]
    color_cropped = slide[y_start:y_end,x_start:x_end]
    
    # tissue percentage
    threshold = 0.9

    count = 0
    
    # save tile cordinates
    size = int(img_cropped.shape[0]/step * img_cropped.shape[1]/step)
    tile_cordinates = np.zeros((size,2))
    arrays = np.array([])

    for i in range(int(img_cropped.shape[0]/step)):
        for j in range(int(img_cropped.shape[1]/step)):
            mask_sum = img_cropped[i*step:i*step+step , j*step:j*step+step].sum()
            mask_max = step * step * 255
            area_ratio = mask_sum / mask_max
            
            if area_ratio > threshold:
                tile = color_cropped[i*step:i*step+step , j*step:j*step+step]
                x_ = j*step
                y_ = i*step
                tile_cordinates[count,0]=x_
                tile_cordinates[count,1]=y_
                 
                arrays = np.save(save_dir + filename[:-4] + "_" + str(count), tile)
                
#                 uncomment these if .png-images of the tiles are necessary
                image_color = Image.fromarray(tile)
                image_color.save(save_dir + filename + str(count) + "_col.png", "PNG") 
                count+=1

    return tile_cordinates, arrays
