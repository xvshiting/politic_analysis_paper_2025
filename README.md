# politic_analysis_paper_2025

## Data

All raw data are stored in the `data` directory.

>
The "Policy Content" in the raw data contains policy documents issued by the Chinese central government and the Shandong provincial government. This study conducted text preprocessing on the contents in this column to create a corpus of policy textual data. We have already translated other fields in the data files into English, but the "Policy Content" has remained in its original Chinese form, as translating it might affect the results of the policy topic modeling.

## Code

We have three code files:`province_analysis.py` 、 `country_analysis.py`  and
`utils.py` all stored in root directory.

- utils.py: contains common use functions, such as: init_dir, loading stop words.
- country_analysis.py: contains code analyze country policy materials.
- province_analysis.py: contains code analyze Shandong Province policy materials. 

This repo also have two ipynb files:`province.ipynb` and `country.ipynb`, they are our intermediate process product which contains code and images.
These two ipynb files should be runing under jupyterlab or notebook editors.

## output

Some vital results are stored in `output` directory.

## others 

- stopwords:`stopwords.txt`
- font:`SimSun.ttf`
- word cloud picture: `chinese.png` and `shandong.png`

These files are all under root directory.
