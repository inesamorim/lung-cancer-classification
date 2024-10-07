import pylidc as pl
import numpy as np
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