# README

This is a project aimed at determining the subtypes of malignant tumours in the breast region. It uses the Breast Cancer MRI dataset from Duke University as the base and creates machine-learning based classifiers using features extracted by PyRadiomics, a python library aimed at extracting tumour characteristics using radiomics.

## Directory structure

1. segmentation:
  a. convert_nifti.ipynb: converts DICOM scans into NIFTI
  b. segmenting and pyradiomics extraction.ipynb: segments lesions from breast NIFTI files as per annotations specified by the dataset. Extracts features for pre and post contrast sequences using PyRadiomics.
2. modelling: contains files for modelling with and without feature scaling, and for feature importance teseting
3. Visualisation: contains Tableau workbooks connected to Google Drive for performance and data visualisation.
