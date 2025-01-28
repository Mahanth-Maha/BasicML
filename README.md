# BasicML
Learning basics of Machine learning 

Language : Python - numpy, matplotlib, seaborn, scikit-learn, **pytorch**


## Folder structure
MahaML
---
My implementation of algorithms *with out using any in built modules* and only depending on `numpy` , `matplotlib`, etc

'TYPE' _basics (folders ending with _basics)
---
Learning of algorithms experimenting with datasets, implementations, simulations, etc 
Also learning how to use libraries, and implement similar of my own in MahaML

kaggle
---
competitions, submissions and related files

datasets
---
some toy datasets to test 


## My Useful Setup codes

OS : Windows (Cmd/PS)

### Setup a venv and create a kernal for vscode 

```bash
conda create --name <NAME> python==<PyVersion>
y
conda activate <NAME>
conda install -c anaconda ipykernel
y
python -m ipykernel install --user --name <NAME> --display-name <"Kernal_Name">
conda deactivate
exit

```



- change ```<NAME>``` with env-name (EX: torchEnv)
- change ```<"Kernal_Name">``` with env-name (EX: "trenv" , note:in string quotes)

### To set jupyter notebook to print all / last expressions

```python
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "last"  # Options: 'last', 'all', 'last_expr'
```

[![Page Views Count](https://badges.toozhao.com/badges/01JHW6EJD8WR0PK30ZESDZRPDT/green.svg)](https://badges.toozhao.com/stats/01JHW6EJD8WR0PK30ZESDZRPDT "Get your own page views count badge on badges.toozhao.com")