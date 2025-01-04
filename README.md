### Requirements

Install the required libraries
```bash
pip install -r requirements.txt
```

---

### Running the Code

To train or evaluate the model:

```bash
python main.py
```

---

### Configuration in main.py

1. **Specify Checkpoints Directory**:
   - In `main.py`, set the directory where checkpoints will be saved, default is ```./experiments```

2. **Choose Loss Function**:
   - On **line 64** of `main.py`, select the loss function:
     - `crossentropy`: Use GeoCLIP loss.
     - `triplet`: Use Triplet loss.
     - `entropywithdis`: Our strategy.

3. **Save Checkpoint for GeoCLIP**:
   - In `geoclip.py`, specify the checkpoint to use


### Streetscape dataset download
run code ```download_jpegs.py```
```bash
python run ./download_imgs/download_jpegs.py
```

Detailed description refer to https://github.com/ualsg/global-streetscapes/wiki/2-Download-images


### osv5m dataset download

Refer to https://huggingface.co/datasets/osv5m/osv5m
