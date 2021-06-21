import mir.multiresolutionimageinterface as mir #requires python 3.6.
import openslide
import numpy as np
import os
from PIL import Image

#=====================================================================================
# process_tumor_tiles() runs the whole tiling script and saves the tiles to directory
#=====================================================================================

# reads WSI-files and XML-annotations, converts them to numpy array, tiles the 
# annotated area and saves tiles to directory
# input: orig_src   = tumor WSI-directory,
#        masks_dir  = directory to save the masked WSIs to
#        xml_src    = lesion annotations directory
#        save_dir   = directory to save the tiles to
#        level      = level from which the WSIs are read
#        tile_size  = wanted tile size

# mask_annotations-code from: https://camelyon17.grand-challenge.org/Data/


def process_tumor_tiles(orig_src, masks_dir, xml_src, save_dir, level, tile_size):
  
    for filename in os.listdir(orig_src):
        if filename.endswith(".tif"):
            mask_annotations(orig_src, xml_src, save_dir, filename)
            img = openslide.OpenSlide(os.path.join(masks_dir, filename))  
            img_mask = np.array(img.read_region((0, 0), level, img.level_dimensions[level]))
            img = openslide.OpenSlide(os.path.join(orig_src, filename))  
            img_orig = np.array(img.read_region((0, 0), level, img.level_dimensions[level]))
            tiles = tile_im(img_mask, img_orig, tile_size, filename, save_dir)[1]
        else:
            continue
    return tiles

def mask_annotations(orig_src, xml_src, save_dir, filename):

    output_path = (os.path.join(save_dir, filename))
    reader = mir.MultiResolutionImageReader()
    mr_image = reader.open(os.path.join(orig_src, filename))
    annotation_list = mir.AnnotationList()
    xml_repository = mir.XmlRepository(annotation_list)
    xml_repository.setSource(os.path.join(xml_src, filename[:-3]+"xml"))
    xml_repository.load()
    annotation_mask = mir.AnnotationToMask()
    camelyon17_type_mask = False
    label_map = {'metastases': 1, 'normal': 2} if camelyon17_type_mask else {'_0': 1, '_1': 1, '_2': 0}
    conversion_order = ['metastases', 'normal'] if camelyon17_type_mask else  ['_0', '_1', '_2']
    annotation_mask.convert(annotation_list, output_path, mr_image.getDimensions(), mr_image.getSpacing(), label_map, conversion_order)
    
    return mr_image


# masks and tiles the masked image (array) to smaller tiles according to step size
# input: image array, step size, filename, save directory
# output: array of tile cordinates, saved arrays .npy-format

def tile_im(img_mask, img_orig, step, filename, save_dir):
    
    m,n = img_orig.shape[0], img_orig.shape[1]
    tissue_mask = img_mask[:, :, 1]

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
    color_cropped = img_orig[y_start:y_end,x_start:x_end]
    
    # tumor tissue percentage
    threshold = 0.9

    count = 0
    
    # save tile cordinates
    size = int(img_cropped.shape[0]/step * img_cropped.shape[1]/step)
    tile_cordinates = np.zeros((size,2))
    arrays = np.array([])

    for i in range(int(img_cropped.shape[0]/step)):
        for j in range(int(img_cropped.shape[1]/step)):
            mask_sum = img_cropped[i*step:i*step+step , j*step:j*step+step].sum()
            mask_max = step * step
            area_ratio = mask_sum / mask_max
            
            if area_ratio > threshold:
                tile = color_cropped[i*step:i*step+step , j*step:j*step+step]
                x_ = j*step
                y_ = i*step
                tile_cordinates[count,0]=x_
                tile_cordinates[count,1]=y_
                 
                arrays = np.save(save_dir + filename + "_" + str(count), tile)
                
#                 uncomment these if .png-images of the tiles are necessary
                image_color = Image.fromarray(tile)
                image_color.save(save_dir + filename + str(count) + "_col.png", "PNG") 
                count+=1

    return tile_cordinates, arrays
