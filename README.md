# This is a pre-alpha POC (Proof Of Concept)

## Instructions:
clone:
```
git clone https://github.com/SicariusSicariiStuff/X-Ray_Vision.git
```

Settings up venv, (tested for python 3.11, probably works with 3.10)
```
python3.11 -m venv env
source env/bin/activate
```

Install dependencies
```
pip install git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3
pip install torch
pip install pillow
pip install accelerate
```

# Running inference

Usage:
```
python xRay-Vision.py /path/to/model/ /dir/with/images/
```
The output will print to console, and export the results into a dir named after your image dir with suffix "_TXT"

So if you run:
```
python xRay-Vision.py /some_path/x-Ray_model/ /home/images/weird_cats/
```
Then results will be exported to:
```
/home/images/weird_cats_TXT/
```
