#useful functions

import pylidc as pl
import numpy as np
from scipy.spatial.distance import cdist
from typing import List, Literal
from pylidc.utils import consensus
import matplotlib.pyplot as plt
import os
from pathlib import Path



def get_nodule_class(anns: List[pl.Annotation]) -> Literal[-1, 0, 1]:
    malign = round(np.mean([ann.malignancy for ann in anns]))
    return get_malignancy_class(malign)

classes = [0, -1, -1, 0, 1, 1]
def get_malignancy_class(malign: int) -> Literal[-1, 0, 1]:
    """
    1 - Sus ඞ
    0 - ¯\_(ツ)_/¯  
    -1 - no sus
    """

    return classes[malign]

def filename_from_annotation(ann: pl.Annotation) -> str:
    return ann.scan.patient_id + "-" + str(ann.id)

#########################################################################################################

def _group_nodules_by_proximity(annotations: List[pl.Annotation], distance_threshold: float):
    """ Group nodules manually based on spatial proximity using centroid distance. """
    
    centroids = []
    # Calculate centroids for each nodule
    for ann in annotations:
        centroid = np.array(ann[0].centroid)
        #print(f"Centroid: {centroid}")
        centroids.append(centroid)
    # Calculate distances between all pairs of centroids
    distances = cdist(centroids, centroids)
    
    # Group nodules based on a distance threshold
    groups = []
    visited = set()
    
    for i, centroid in enumerate(centroids):
        if i in visited:
            continue
        
        # Start a new group with the actual annotation, not the index
        group = [annotations[i][0]]
        visited.add(i)
        
        # Find all nodules within the distance threshold
        for j in range(len(centroids)):
            if i != j and distances[i, j] <= distance_threshold:
                group.append(annotations[j][0])  # Append the actual annotation instead of the index
                visited.add(j)
        
        groups.append(group)
    
    return groups

def cluster_annots(scan: pl.Scan, distance_threshold: float=10.0, verbose: bool=False) -> List[List[pl.Annotation]] | None:
    mprint = print if verbose else lambda *_: None
    patient_id = scan.patient_id

    if len(scan.annotations) == 0:
        mprint("Patient %s has no annotations" %(scan.patient_id))
        mprint("################################################")
        return None

    try:
        # Try to cluster annotations automatically
        nodules = scan.cluster_annotations()
        
        # Check if the clustering produced usable results (at least one group of nodules)
        if len(nodules) == 0:
            raise ValueError("Automatic clustering failed, falling back to manual grouping.")

    except Exception as e:
        # If automatic clustering fails, fallback to manual grouping
        mprint(f"Error with automatic clustering for patient {patient_id}: {e}")
        
        # Use manual grouping based on proximity
        mprint("Patient %s has %d annotations" %(scan.patient_id, len(scan.annotations)))
        annotations = [[annotation] for annotation in scan.annotations]  # Create individual nodule lists
        nodules = _group_nodules_by_proximity(annotations, distance_threshold)
        mprint("################################################")
    
    return nodules



#######################################################################################################################

def zoomout_nodule(bbox, width, height, n_slices):
    # TODO: what if is outside the picture?
    x_acres = (width - (bbox[0].stop - bbox[0].start))//2
    y_acres = (height - (bbox[1].stop - bbox[1].start))//2

    x_slice = slice(bbox[0].start - x_acres, bbox[0].start - x_acres + width, None)
    y_slice = slice(bbox[1].start - y_acres, bbox[1].start - y_acres + width, None)

    z_slice = np.linspace(bbox[2].start, bbox[2].stop, n_slices, endpoint=False, dtype=int)

    return (x_slice, y_slice, z_slice)

def get_cropped_annot(scan: pl.Scan, nods, mask: bool=False):
    if len(scan.annotations) == 0:
        return None

    vol = scan.to_volume(verbose=False)

    images = []
    masks = []
    for anns in nods:
        cmask, cbbox, _ = consensus(anns, clevel=0.5,
                                pad=[(10,10), (10,10), (0,0)])

        full_mask = np.full_like(vol, False, dtype=bool)
        full_mask[cbbox] = cmask

        #cbbox = zoomout_nodule(cbbox, MAX_NODULE_WIDTH, MAX_NODULE_HEIGHT, N_SLICES)
        cropped = vol[cbbox]
        cropped_mask = full_mask[cbbox]
        if mask:
            cropped[cropped_mask] = -1000 # TODO: maybe pick different value

        # Scale values
        # TODO: maybe pick different values
        cropped += 1000
        cropped = cropped / 2000

        images.append(cropped)
        masks.append(cropped_mask)

    return images, masks