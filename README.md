# Transformer for Molecular Property Prediction: Lessons Learned from the Past Five Years

This repo provides the data and the code used to generate the figures and tables in the corresponding review article:
* [Arxiv](https://arxiv.org/abs/2404.03969)

## Project requirement
The project is compatible with Python version >= 3.11, and the packages are managed by `poetry`. So, you need to have 
`poetry` installed on your device. You can do so by running this command

``` shell
curl -sSL https://install.python-poetry.org | python3 -
```

For more information, refer to the [documentation page](https://python-poetry.org/docs/#installing-with-the-official-installer) 


To install the project dependencies, run

```shell
poetry install --no-root
```

## Execution
To reproduce the figures and the tables, simply run

```shell
python3 main.py
```
The `main.py` file calls the corresponding functions from the `scripts` folder. These functions access their data 
from the `data` folder. The figures and tables will be saved to the locations as shown in the printed messages after 
excution.

You can also check the `reproduce_figures.ipynb` file for visualising the figures directly.

## Supplementary info

In table 6, we mention that some of the information are extracted from the provided code of the papers as the info was 
not stated in the original manuscript. The `permalink` for each of these values are provided at the 
`data/num_parameters/info_extracted_from_code.csv`file.