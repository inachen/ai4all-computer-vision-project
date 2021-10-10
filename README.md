# UCSF AI4ALL Computer Vision for Covid-19 Chest X-Ray Classification

Materials for the AI4All chest x-ray project (July 2020) [1]

## Notebooks
Coding notebooks used for guided coding during the first week

- Day x notebooks: notebooks for the first week of guided analysis
- Extra code notebooks: starting code for follow-up exploration
- class_demo_code: notebooks from live coding

## Data

Data used for this project were curated from two publicly available datasets
- Cohen dataset [2]
- Stanford dataset [3]

Note that data are not provided with this repository and should be obtained by accessing the datasets.

### Data split

After obtaining appropriate access to the datasets, use the script below to subsample and split the data, as well as generating the associated metadata files.

`utils/create_class_dataset.py`

- From Cohen dataset: Frontal view images of Covid-19 and No Finding cases
- From CheXpert dataset: Randomly selected subset of No Finding images 
- Split data 90-10 into train dataset (provided to students) and test dataset
    + train data location: data/images
    + test data location: data/test

### Dataset provided to students

Training set of chest x-ray images. Frontal views (AP, PA, AP supine) of No Finding and Covid cases were selected from Cohen dataset. A small random subset of frontal views of No Finding cases were selected from the Stanford CheXpert dataset.

- images/nofinding: chest x-ray images with no diagnosis
- images/covid: chest x-ray images for patients with confirmed SARS-Cov-2 diagnosis
- test/nofinding,covid: test image set, no overlapping patients with train set
- metadata_test.csv: metadata information for test images
- metadata_train.csv: metadata information for training images

## Docs

Additional metadata files and documents. The following files were provided to students for data exploration and can be obtained from the respective datasets

- cohen_schema.md: metadata annotation for the Cohen dataset
- cohen_severity_scores.csv: severity scores for images in the Cohen dataset
- cohen_predictions.csv: predicted observations added to the metadata frame (predictions from torchxrayvision pre-trained neural network)
- metadata_cohen_all.csv: additional metadata for all images in Cohen dataset (includes non-Covid pneumonias and lung illnesses)
- metadata_cohen_covid.csv: additional metadata for Covid cases in Cohen dataset
- metadata_cohen_nofind.csv: additional metadata for No Finding cases in Cohen dataset
- metadata_stanford_all.csv: additional metadata for all images in Stanford CheXpert dataset (includes non-Covid pneumonias and lung illnesses)
- metadata_stanford_nofind.csv: additional metadata for No Finding cases in Stanford CheXpert dataset
    
## References

[1] AI4All paper
[2] Cohen JP, Morrison P, Dao L. COVID-19 Image Data Collection. ArXiv200311597 Cs Eess Q-Bio [Internet]. 2020 Mar 25
[3] Irvin J, Rajpurkar P, Ko M, Yu Y, Ciurea-Ilcus S, Chute C, et al. CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison. Proc AAAI Conf Artif Intell. 2019 Jul 17;33(01):590–7.

