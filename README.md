# shapley_mssc_6250
This repository contains the  code submitted for MSSC 6250: Statistical Machine Learning course project

Contributors: 
- Brigida Zhunio Cardenas
- David Aguilera
- Dibaloke Chanda

# Instruction to run the Google Colab Notebook

We strongly to the code using the following colab notebook link. 


[MSSC 6520 Google Colab Notebook](https://colab.research.google.com/drive/15s_ESfI1kevsTw_YhuJV5adFHwKLtmwL?usp=sharing)


# Instructions to Run Python Code in "Notebooks" Folder (Locally)

### Clone the repository

First clone the repository with Github desktop or through a CLI with the following command:

```bash
git clone https://github.com/dibalokechanda/shapley_mssc_6250.git
```

### Package Installation Manually 

For the code implementation, you need install the following libraries
 - numpy
 - pandas
 - xgboost
 - Sckit-learn
 - shap

> We strongly recommend creating a virtual enviroment beforing installing them.

 The command the install these:
```bash
pip install shap numpy pandas xgboost 
pip install -U scikit-learn
```

### Package Installation via requirement .txt
- You need to install the packages mentioned in the requirments.txt file.
Make sure you are in the root of the cloned repo and run the following command:

```bash
pip install -r requirements.txt
```

# Description of the contents

### Lasso
Contains code to perform Lasso regression with simulated data.

### Notebooks 
Contains code implementing the [SHAP](https://shap.readthedocs.io/en/latest/) library.

