# Image Rotation Correction

## Installation

Using Python version 3.8
```
conda create --name cse455 python=3.8
conda activate cse455
```
Install [PyTorch](https://pytorch.org/) using instructions from their website. The most recent version of PyTorch at the time of this project is 1.13.1
```
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```
Install the other packages using the requirements file
```
pip install -r requirements.txt
```

## Run the demo script
```
python src\demo.py --model ./experiments/run005/ --files ./data/rotated_birds/00a0c41bf96a42778ac09cfa19989a3b.jpg --output ./output
```