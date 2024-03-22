# Data collection

This folder contains the data to support some analysis in the manuscript. In which we detail the following:

## `multi_task_data` 
This folder contains the `tox21`, `ClinTox`, and `SIDER`datasets as obtained from MoleculeNet. 

These files were used to generate Supplementary tables 11, 12, and 13, which show the percentage of positive class per 
task.

## `num_parameters`
This folder contains the information needed to calculate the number of parameters. Most of the information were 
explicitly mentioned in the corresponding papers. For the few that weren't, the file 
`num_parameters/info_extracted_from_code.csv` shows the code lines where the missing information were extracted. 

These data are the base for table 6 and part of table 5.

## `objectives`

This folder contains the ablation analysis for the MolBERT, K-BERT, and MAT  [^1]. 

These data are used to generate Figure 6 and Supplementary Figures 8 and 9.

## `Performance_comparison`
This folder contains the reported performance metrics for the transformer models as well as the compared classical ML 
and DL models [^1].

These data are used to generate Figure 2 and Supplementary Figure 7.

## `Pretrain_dataset_size`
This folder contains performance values from ChemBERTa-2 and MolFormer which correspond to models trained with different pre-training 
dataset sizes [^1] .

These data are used to generate Figure 4.

## `representation`
This folder contains performance values from MAT and Mol-BERT which compared their performance against other 
(reimplemented) transformer models [^1]. 

These data are used to generate Figure 5.



## Footnotes
[^1]: The data in corresponding 
publications were available only as PDF format. The information were extracted by taking a screenshot of the 
corresponding tables and using the "Data from Picture" Excel function. When the output of the function was not helpful, 
manual extraction was done.

