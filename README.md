# Audio MNIST classification
This project is a simple audio classification using the Audio MNIST dataset. The dataset consists of 30,000 audio samples of spoken digits (0-9) in English.

### Setup

#### Pre-requisites
- [MiniConda](https://docs.conda.io/en/latest/miniconda.html) installed on your system

#### Create Conda Environment
1. Create a new conda environment
```bash
conda env create --name audio-mnist python=3.8.10
```
2. Activate the environment
```bash
conda activate audio-mnist
```
3. Install tensorflow
```bash
pip install 'tensorflow==2.13.0'
```
4. Install other dependencies
```bash
pip install -r requirements.txt
```

### Dataset
Demo data can be found in the `data_demo` directory and it's already packed with the repository.

But if you want a full dataset (e.g. for training purpose), you can find it [here](https://www.kaggle.com/datasets/sripaadsrinivasan/audio-mnist/data).Data should be unzipped into the `data` directory.

To prepare it for training, run the following command:
```bash
python3 split_data.py
```

### Notebooks
- `train.ipynb`: Training the model, saving the model and evaluating it
- `demo.ipynb`: Demo of the model, loading the model from `saved_models` and predicting on `data_demo` samples

