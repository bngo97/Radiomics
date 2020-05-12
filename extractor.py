import os
import sys
import numpy as np
import pydicom as dicom
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk
import six

from tqdm import tqdm
from radiomics import featureextractor
from pandas.io.json import json_normalize


rootdir = sys.argv[1]

# Collect paths to all .dcm files
dcm_files = []
for root, subdirs, files in os.walk(rootdir):
    for file in files:
        if '.dcm' in file:
            dcm_files.append(os.path.join(root, file))

# Extract features
data = []
extractor = featureextractor.RadiomicsFeatureExtractor('params.yaml')
feature_names = ['firstorder', 'shape2D']

otsu_filter = sitk.OtsuThresholdImageFilter()
otsu_filter.SetInsideValue(0)
otsu_filter.SetOutsideValue(1)

for file in tqdm(dcm_files):
    image_path = file
    ds = dicom.dcmread(image_path)
    img = sitk.GetImageFromArray(ds.pixel_array)
    img_255 = sitk.Cast(sitk.RescaleIntensity(img), sitk.sitkUInt8)

    # Segmentation via Otsu Thresholding
    #seg = sitk.BinaryThreshold(img_255, lowerThreshold=100, upperThreshold=400, insideValue=1, outsideValue=0)
    seg = otsu_filter.Execute(img_255)
    # Extract features
    result = extractor.execute(img_255, seg)

    # Create a dict of useful values
    row = {}
    row['name'] = file
    for key, value in six.iteritems(result):
        for feature in feature_names:
            if feature in key:
                row[key[9:]] = float(value)

    data.append(row)

df = json_normalize(data)
df.to_csv('{}-data.csv'.format(rootdir), index=False)
