# BasicML
Learning basics of Machine learning 

Language : Python - **numpy**, matplotlib, seaborn, scikit-learn, **pytorch**



This repository is a comprehensive portfolio documenting my journey into Machine Learning, from foundational theory to practical application. It showcases a commitment to understanding algorithms from first principles and applying them to solve real-world problems.

The core of this repository is **MahaML**, a custom Python package where I have implemented major machine learning algorithms from scratch using only NumPy. This project was undertaken to build a deep, mathematical understanding of the models that power modern data science.



## Setup and Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/Mahanth-Maha/BasicML.git
cd BasicML
```

### Step 2: Create and Activate Environment

Using Conda:

```bash
conda create --name basicML python==3.9 -y
conda activate basicML
conda install -c anaconda ipykernel -y
python -m ipykernel install --user --name basicML --display-name "BasicML"
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Step 3: Launch Jupyter

```bash
jupyter lab
```

### Optional – Display All Output Cells

In a notebook:

```python
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
```

## Folder Overview

### `MahaML/`

A self-written **ML algorithms package** using only `numpy` and `matplotlib`.
Each algorithm is implemented from first principles for a complete mathematical understanding.

Includes:

* Regression and classification models
* Ensemble methods (Bagging, Boosting, Stacking)
* Clustering and dimensionality reduction prototypes
* Helpers and testing utilities

Example:

```bash
python MahaML/linearRegression.py
```

### `ML_NLP_basics/`

Contains notebooks demonstrating:

* Exploratory Data Analysis (EDA)
* Text preprocessing
* Supervised and unsupervised ML workflows
* Building algorithms step-by-step and testing on datasets

Datasets include: IMDb reviews, spam mails, student grades, housing data, and more.

### `python_basics/`

Notebooks covering:

* Python data types and structures
* User-defined functions and modules
* Plotting and visualization
* Probability and linear algebra essentials

### `pytorch_basics/` and `DL_basics/`

Early work in deep learning.
Focus on understanding tensor operations, autograd, and simple feedforward architectures.

### `kaggle/`

Contains scripts, data, and submissions from Kaggle competitions such as:

* **Titanic Survival Prediction**
* **IMDB Genre Classification**
* **Playground Series Experiments**

### `Projects/` (Keeping them in local - in progress)

Applied ML and DL projects demonstrating practical problem-solving:

* CTR (Click-Through-Rate) Prediction
* News Recommendation System (EBNeRD dataset)
* Online Shoppers Purchase Intention Analysis

### `Courses/`

Lecture slides and references from **Neural Networks for Machine Learning** and other learning materials.

### `Notes/`

Self-written markdown summaries and formula sheets for quick reference.


## `MahaML` Library: Implementation Progress

This section tracks the development status of the from-scratch algorithm implementations within the `MahaML` package.

### Supervised Learning

  * **Regression**
      * [x] Linear Regression
      * [x] k-Nearest Neighbors (k-NN)
  
  * **Classification**
      * [x] Logistic Regression
      * [x] k-Nearest Neighbors (k-NN)
      * [x] Naive Bayes
      * [x] Support Vector Machine (SVM)
      * [x] Decision Tree
  * **Ensemble Methods**
      * [x] Bagging (General + Random Forest)
      * [x] Boosting
      * [x] Stacking

### Unsupervised Learning

  * **Clustering**
      * [ ] k-Means Clustering (in progress)

### Future Development Roadmap

  * [ ] Clustering - DBScan, Hierarchical
  * [ ] Dimensionality Reduction (e.g., PCA, T-sne)
  * [ ] Recommender System Algorithms

## Example Workflow

1. Start by reviewing foundational notebooks in `python_basics/`.
2. Explore algorithm implementations in `MahaML/` — for instance, `linearRegression.py`.
3. Compare your implementation’s output with `scikit-learn` equivalents in `ML_NLP_basics/`.
4. Progress into neural networks using notebooks under `pytorch_basics/`.
5. Apply knowledge to real datasets in the `kaggle/` and `Projects/` directories.



### Technologies Used

* Python 3.9+
* NumPy, Matplotlib, Seaborn
* scikit-learn
* PyTorch
* NLTK, pandas
* Jupyter Notebook / VSCode

## License

This repository is open for educational and learning purposes.
Feel free to fork, explore, or contribute improvements with proper attribution.


# My Useful Setup codes (not related to this repo)

OS : Windows (Cmd/PS)

### Setup a venv and create a kernal for vscode 

```bash
conda create --name <NAME> python==<PyVersion> -y
conda activate <NAME>
conda install -c anaconda ipykernel -y
python -m ipykernel install --user --name <NAME> --display-name <"Kernal_Name">

pip install requirements.txt
```

- change ```<NAME>``` with env-name (EX: torchEnv)
- change ```<"Kernal_Name">``` with env-name (EX: "trenv" , note:in string quotes)

### To set jupyter notebook to print all / last expressions

```python
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "last"  # Options: 'last', 'all', 'last_expr'
```

## Contact

**Author:** $\large \textrm{Mahanth Yalla (Maha)}$, MTech AI, IISc Bangalore

**Focus Areas:** Machine Learning, Deep Learning - CV & NLP

**Portfolio:** [mahanthyalla.in](https://mahanthyalla.in)

**GitHub:** [github.com/Mahanth-Maha](https://github.com/Mahanth-Maha)
