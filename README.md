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
conda create --name <NAME> python==<PyVersion> pandas numpy ipykernel ; 
conda activate <NAME> ; 
conda install -c anaconda ipykernel ; python -m ipykernel install --user --name <NAME> --display-name <"Kernal_Name"> ; 
conda deactivate ; 
exit
```

- change ```<NAME>``` with env-name (EX: torchEnv)
- change ```<"Kernal_Name">``` with env-name (EX: "tr" , note:in string quotes)

### To set jupyter notebook to print all / last expressions

```python
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "last"  # Options: 'last', 'all', 'last_expr'
```