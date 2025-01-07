# BasicML
Learning basics of machine learning 

Language : Python


MahaML
---
My implementation of algorithms with out using any in built modules and only depending on `numpy` , `matplotlib`, etc

'TYPE' _basics
---
Learning of algorithms experimenting with datasets, implementations, simulations, etc 
Also learning how to use libraries

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
- change ```<"Kernal_Name">``` with env-name (EX: "tr" , note:in string quotes)

Torch env :
```bash
conda create --name torch python==3.10
y
conda activate torch 
conda install -c anaconda ipykernel 
y
python -m ipykernel install --user --name torch --display-name "torch" 
conda deactivate 
exit
```

### To set jupyter notebook to print all / last expressions

```python
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "last"  # Options: 'last', 'all', 'last_expr'
```