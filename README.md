# inf889e-project
This is my project developed as part of the course *INF889E - Méthodes d'intelligence artificielle en bioinformatique* at *Université du Québec à Montréal*.

The project consists of an analysis of HIV sequences by geographical areas (countries, regions, continents) with the aim of detecting the most decisive motifs for classifying a genome in a particular area, and therefore the motifs that might be specific to certain areas.

## Prerequisites
### Import functions
Before trying to run the script or notebook, you need to install the required dependencies listed at the beginning of the script or notebook, usually with `pip` or `conda`. On a fresh installation of Python, you will probably need to run the following:
```
pip install joblib
pip install numpy
pip install pandas
pip install progressbar2
pip install bio
pip install sklearn
pip install mpl_toolkits
pip install dna_features_viewer 
pip install matplotlib
pip install seaborn
```
Note that *Biopython* also requires Visual C++ Build Tools.
### Data
Data samples are already included in the source code, in the folder `data`. 
If you want to use your own data:
* Download one *FASTA* file per region: the filename will be the target class
    * You can also get only one file and later chose to use the countries as the target classes
* They must have the default Los Alamos HIV Database header, which is: `>Subtype.Country.Year.Name.Accession`

The three datasets provided are described below.

#### `complete` set
This set contains all complete unaligned genomes available on the Los Alamos HIV Database, with one file for each region. There are huge disparities between regions, ranging from more than 5000 records for North America to about 40 for the Caribbean or Oceania. In addition, there are no sequences from Central America.

#### `mixed` set
This set contains all the previously mentioned files that include more than 600 records, but a more flexible version of the others regions, tolerating some non-complete genomes to make up for the disparities. This allows for the inclusion of Central America, and a more evenly distributed dataset to be processed.

#### `aligned` set
This set contains all complete aligned genomes available on the Los Alamos HIV Database, with one file for each region. While the first idea of the projet was to work with unaligned sequences, a lot of the work started unintentionally on these aligned sequences, so the data and the results are kept here for comparaison.

## Important variables
At the beginning of the script, there are some variables you may want to have a look at:
### `scope`
This controls the behaviour of the script. If the scope is set to `ALL`, it will classify the data by region, using the filename of each file in the path folder as the target class. 
If this variable is set to any filename in the path folder, it will categorise the data by country, looking for the target classes in the records.
### `path`
This is the path folder mentioned above. You will probably not have to change this if you use the provided data.
### `model_name`
This will be the file name of the trained model when the scrip will save it (in the root folder).
### `n_samples`
If you want to test the script on a small number of records, you can set this variable to a small number (e.g. 100), and the script will set this number as the maximum number of records it will extract from the data files (after shuffling) for each target.
### `freq`
If this flag is set to true, the script will use the frequency of motifs appearing in the sequences instead of their sum.
### `step`
This is the elimination step variable for the Recursive Feature Elimination (RFE).
### `n_features`
This is the number of features to select for the Recursive Feature Elimination (RFE).
### `split_raito`
This is the ratio the script will use when splitting data between train and test data sets.
### `n_components`
This is the dimension that plots will use when visualising train data (2D or 3D)
### `max_incorrect`
This is the maximum number of incorrectly classified records to analyse at the end of the script. As this analysis will show several plots per incorrect record, you may want to use a small number for this one.
### `max_correct`
This is the maximum number of correctly classified records that will be compared to incorrect records in the last step of the final analysis. This has been setup to avoid hours long runs when using a large amount of data.
### `k`
This is the size of the k-mers used as the features.

## Results
The results section contains all the already prepared notebooks for each relevant scope and dataset that has been run for the purpose of this project. Note that the `all` dataset is not available in this repository as it is way too large, but the results are still presented here.