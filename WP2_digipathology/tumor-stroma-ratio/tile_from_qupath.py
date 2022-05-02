#=====================================================

#           Tile from QuPath-annotations

#=====================================================

# Reads QuPath-annotation instances, utilizes paquo-library: 
# https://github.com/bayer-science-for-a-better-life/paquo

# Tiles the annotation sites, plots 10 examples from each class

# required variables:

# project_path = path to QuPath-project
# WSI_path     = path to WSI-files
# tiles_path   = path where to save tiles, should include folders for all classes
# tile_size    = int, width (and height) of the tile
# overlap      = int
# percentage   = percentage of annotation shape required to be in the tiling area in order to make the tile


# running the tile_qupath-function runs the script

from pathlib import Path
from paquo.projects import QuPathProject as QP
from shapely.geometry import box
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import openslide
import numpy as np
import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import random

def tile_qupath(project_path, WSI_path, tiles_path, tile_size, overlap, percentage):

    with QP(project_path, mode='r') as qp:

        for image in qp.images:

            annotations = image.hierarchy.annotations
            name = image.image_name
            
            print("Tiling from " + name)
            print("Number of annotations: "+str(len(annotations)))
            print()
            
            img_os = openslide.OpenSlide(os.path.join(WSI_path, name))
            
            x, y = image.width, image.height
            
            img_os_small = np.array(img_os.read_region((0, 0), 7, (img_os.level_dimensions[7])))
            plt.imshow(img_os_small)
            plt.show()
            
            labels = ["Other", "Stroma", "Tumor"]

            for annotation in annotations:

                label = annotation.path_class.name
                polygon_ = annotation.roi
                tile_count = tile_annotations(x,y,polygon_, img_os, name, label, tile_size, overlap, percentage)
                print(label+": "+str(tile_count)+ " tiles done")

    plot_examples(10, tiles_path, labels)


def tile_annotations(width, height, polygon, img, name, label, size, overlap, percentage):
    
    count = 0
    step = size-overlap

    for i in range(0,int(width-size),step):
        for j in range(0, int(height-size), step):

            tile = box(i,j,i+size,j+size)

            if tile.intersects(polygon):
                
                intersection = tile.intersection(polygon)
                area = intersection.area/tile.area
                
                if area > percentage:
                    img_np = np.array(img.read_region((i, j), 0, (size,size)))
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
                    img_ = Image.fromarray(img_np)
                    img_.save(tiles_path + label+ "/" + name + "_" + label + "_" + str(count) + ".jpeg", "JPEG")
                    count+=1
                    
    return count
