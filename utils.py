import pylidc as pl
import numpy as np
from typing import List, Literal


def get_malignacy_class(anns: List[pl.Annotation]) -> Literal[-1, 0, 1]:
    """
    1 - Sus ඞ
    0 - ¯\_(ツ)_/¯
    -1 - no sus
    """

    classes = [0, -1, -1, 0, 1, 1]
    malign = np.mean([ann.malignancy for ann in anns], dtype=int)
    return classes[malign]

# TODO: _nodule_id is weird
def filename_from_annotation(ann: pl.Annotation) -> str:
    return ann.scan.patient_id + "-" + ann._nodule_id


