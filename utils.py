import pylidc as pl
import numpy as np
from scipy.spatial.distance import cdist
from typing import List, Literal


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