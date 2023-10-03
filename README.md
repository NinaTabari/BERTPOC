# BERTPOC

## Setup
You can setup the python dependencies with the following command. It is recommended to do this in a virtual environment.
```
pip install -r requirements.txt
```

## Train the model

The following command will train and save the model. It is currently hardcoded to training with storyline data.
```
python BERTtest.py
```

## Run the model

The following command will load and run the model against the validate.csv file
```
python BERTRun.py
```