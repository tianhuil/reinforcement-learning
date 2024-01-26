# reinforcement-learning
 
## Setup

```py
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training
```py
PYTHONPATH=.:$PYTHONPATH python src/train_logistics.py
tensorboard --logdir ./data/tb # in another terminal
```

## Good Results

Result: -30:
```
./data/model/1706238544-satisfactoriness-tussle  # -28
./data/model/1706242653-jackfish-glecoma # -33
```
