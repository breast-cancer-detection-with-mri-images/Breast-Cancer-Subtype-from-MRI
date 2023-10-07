# Observations about segmentation

- dicom2nifti converts into a different orientation: RAS oriented
- annotation boxes have to be fixed for that:
  - slice => total - slice
  - rows => columns, columns => rows