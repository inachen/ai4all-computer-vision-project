import os
import pandas as pd
import shutil

# add local paths for datasets
path_to_cohen_data = 'path/to/cohen_dataset'
path_to_stanford_data = 'path/to/cohen_dataset'

# where data will be saved
path_to_datasets = 'datasets'

# subfolders that will be created
path_to_class = os.path.join(path_to_datasets, 'data')
path_to_images = os.path.join(path_to_class, 'images')
path_to_covid = os.path.join(path_to_images, 'covid')
path_to_nofind = os.path.join(path_to_images, 'nofinding')

path_to_test = os.path.join(path_to_class, 'test')

def load_cohen_data():
  '''Filter out the covid-19 cases with frontal view images from dataset'''

  df = pd.read_csv(os.path.join(path_to_cohen_data, 'metadata.csv'))
  
  # frontal view annotations
  front_views = ['AP', 'PA', 'AP Supine']
  
  # select for covid and no finidng images with frontal views
  covid_front = df[(df.finding == 'COVID-19') & df.view.isin(front_views)]
  
  nofind_front = df[(df.finding == 'No Finding') & df.view.isin(front_views)]

  print(f'Cohen Covid frontal views: {len(covid_front)} \nCohen no finding frontal views: {len(nofind_front)}')

  # copy files
  for _, row in covid_front.iterrows():

    image_path = os.path.join(path_to_cohen_data, row.folder, row.filename)
    shutil.copy(image_path, path_to_covid)
    
  for _, row in nofind_front.iterrows():

    image_path = os.path.join(path_to_cohen_data, row.folder, row.filename)
    shutil.copy(image_path, path_to_nofind)
  
  # save csv
  covid_front.to_csv(os.path.join(path_to_class, 'cohen_covid.csv'), index=False)
  nofind_front.to_csv(os.path.join(path_to_class, 'cohen_nofind.csv'), index=False)

  return covid_front, nofind_front

def load_stanford_data(n=424):
  '''
  Randomly sample no finding cases to supplement the covid training set
  Covid frontal views: 438 
  No finding frontal views: 14

  Number of no finding images to select: 424
  '''

  df = pd.read_csv(os.path.join(path_to_stanford_data, 'train.csv'))
  
  # select for no finidng images with frontal views
  nofind_front = df[(df['Frontal/Lateral'] == 'Frontal') & df['No Finding']==1]
  nofind_front = nofind_front.sample(n=n)

  print(f'Stanford no finding frontal views: {len(nofind_front)}')

  # copy files
    
  for _, row in nofind_front.iterrows():
    path_components = row.Path.split(os.sep)
    
    # create filename combining the last 3 elements in path (patient, study, view id)
    filename = '_'.join(path_components[2:])

    image_path = os.path.join(path_to_datasets, row.Path)
    output_path = os.path.join(path_to_nofind, filename)
    shutil.copy(image_path, os.path.join(path_to_nofind, filename))

  # save csv
  nofind_front.to_csv(os.path.join(path_to_class, 'stanford_nofind.csv'), index=False)
  
  return nofind_front

def combine_metadata(cohen_covid_df, cohen_nofind_df, stanford_nofind_df):
  '''Combines the metadata information in the same format across the two datasets''' 

  stanford_nofind_df['filename'] = stanford_nofind_df['Path'].apply(lambda x: '_'.join(x.split(os.sep)[2:]))
  stanford_nofind_df['folder'] = 'nofinding'
  stanford_nofind_df['finding'] = 'No Finding'
  stanford_nofind_df['sex'] = stanford_nofind_df['Sex'].apply(lambda x: 'M' if x=='Male' else 'F' )
  stanford_nofind_df['age'] = stanford_nofind_df['Age']
  stanford_nofind_df['view'] = stanford_nofind_df['AP/PA']
  stanford_nofind_df['patientid'] = stanford_nofind_df['Path'].apply(lambda x: x.split(os.sep)[2])
  stanford_nofind_df['dataset'] = 'Stanford'

  cohen_covid_df['dataset'] = 'Cohen'
  cohen_covid_df['folder'] = 'covid'

  cohen_nofind_df['dataset'] = 'Cohen'    
  cohen_nofind_df['folder'] = 'nofinding'
  
  columns = ['patientid', 'sex', 'age', 'view', 'finding', 'dataset', 'folder', 'filename']

  combined_df = pd.concat([cohen_covid_df[columns], cohen_nofind_df[columns], stanford_nofind_df[columns]])

  combined_df.to_csv(os.path.join(path_to_class, 'metadata_combined.csv'), index=False)

def split_by_patient(df, test_ratio=0.1):

  num_test = int(test_ratio * len(df))

  # patient id frequency table
  id_table = df.patientid.value_counts().rename_axis('patientid').to_frame('counts').reset_index()

  # random shuffle
  id_table = id_table.sample(frac=1)

  # take first n rows that add up to test data
  id_table['count_sum']= id_table.counts.cumsum()
  id_subset = id_table[id_table.count_sum <= num_test]
  
  test_df = df[df.patientid.isin(id_subset.patientid)]
  train_df = df[~df.patientid.isin(id_subset.patientid)]

  print(f'Test data number: {len(test_df)} \nTrain data number: {len(train_df)}')

  return test_df, train_df


def split_test_data():
  '''
  Output:
    Split Covid data:
    Test data number: 43 
    Train data number: 395
    Split No Finding data:
    Test data number: 43 
    Train data number: 395
  '''
  df = pd.read_csv((os.path.join(path_to_class, 'metadata_combined.csv')))

  print('Split Covid data:')
  covid_df = df[df.finding == 'COVID-19']
  covid_test_df, covid_train_df = split_by_patient(covid_df)

  print('Split No Finding data:')
  nofind_df = df[df.finding == 'No Finding']
  nofind_test_df, nofind_train_df = split_by_patient(nofind_df)

  train_df = pd.concat([covid_train_df, nofind_train_df])
  test_df = pd.concat([covid_test_df, nofind_test_df])

  # copy images
  for _, row in test_df.iterrows():
    image_path = os.path.join(path_to_images, row.folder, row.filename)
    output_path = os.path.join(path_to_test, row.folder)
    shutil.move(image_path, output_path)

  train_df.to_csv(os.path.join(path_to_class, 'metadata_train.csv'), index=False)
  test_df.to_csv(os.path.join(path_to_class, 'metadata_test.csv'), index=False)


# -----------------
# Select data
# -----------------

cohen_covid_df, cohen_nofind_df = load_cohen_data()
stanford_nofind_df = load_stanford_data()

# -----------------
# Combine metadata
# -----------------

combine_metadata(cohen_covid_df, cohen_nofind_df, stanford_nofind_df)

# -----------------
# Split test data
# -----------------

split_test_data()






