# inf889e-project
This is my project developed as part of the course *INF889E - Méthodes d'intelligence artificielle en bioinformatique* at *Université du Québec à Montréal*.

The project consists of an analysis of HIV sequences by geographical areas (countries, regions, continents) with the aim of detecting the most decisive motifs for classifying a genome in a particular area, and therefore the motifs that might be specific to certain areas.

## Prerequisites
### Import functions
Before trying to run the script or the notebook, you must install the required dependencies listed at the beginning of the script or the notebook, usually with `pip` or `conda`. On a fresh installation of Python, you would probably need to run the following:
```
pip install joblib
pip install numpy
pip install pandas
pip install progressbar
pip install bio
pip install sklearn
pip install mpl_toolkits
pip install dna_features_viewer 
pip install matplotlib
pip install seaborn
```
Note that *Biopython* also requires Visual Studio C++ development tools.
### Data
Data samples are already included in the source code, in the folder `data`. 
If you want to use your own data:
* Download one *FASTA* file per region: the filename will be the target class
    * You can also get only one file and later chose to use the countries as the target classes
* They must have the default Los Alamos HIV Database header, which is: `>Subtype.Country.Year.Name.Accession`

## Important variables
At the beginning of the script, there are some variables you may want to have a look at:
### `scope`
This controls the behaviour of the script. If the scope is set to `ALL`, it will classify the data by region, using the filename of each file in the path folder as the target class. 
If this variable is set to any filename within the path folder, it will classify the data by country, looking for the target classes inside the records.
### `path`
This is the path folder mentioned above. You will probably not have to change this if you use the provided data.
### `model_name`
This will be the filename of the trained model when the scrip will save it (in the root folder).
### `n_samples`
If you want to test the script on a small number of records, you can set this variable to a small number (e.g. 100), and the script will set this number as the maximum number of records it will extract from the datafiles (after shuffling) for each target.
### `freq`
If this flag is set to true, the scrip will use frequency of motifs within the sequences instead of counting them.
### `step`
This is the elimination step variable for the recursive feature elimination (RFE).
### `n_features`
This is the number of features to select for the recursive feature elimination (RFE).
### `split_raito`
This is the ratio the script will use when splitting data in train and test datasets.
### `n_components`
This is the dimension plots will use when visualising train data (2D or 3D)
### `max_incorrect`
This is the maximum number of incorrectly classified records to analyse at the end of the script. Because this analysis will show few plots per incorrect record, you may want to use a small number for this one.
### `k`
This is the size of the k-mers used as the features.