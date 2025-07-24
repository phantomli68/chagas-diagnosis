

## Chagas Diagnosis：PhysioNet Challenge 2025


### Introduction



This project was developed for the **PhysioNet Challenge 2025**.

This repository contains the full pipeline for data preprocessing, model training, evaluation, and submission generation.

### Dataset



We use the **SaMi-Trop** and **PTB-XL** datasets from the **PhysioNet/CinC Challenge 2025**.

- **SaMi-Trop Dataset**: Consists of 1,631 12-lead ECG recordings from confirmed Chagas patients in Brazil, collected between 2011 and 2012. Each recording lasts approximately 7.3 or 10.2 seconds with a sampling rate of 400 Hz. All labels are serologically confirmed positive.
- **PTB-XL Dataset**: Includes 21,799 12-lead ECG recordings collected in Europe between 1989 and 1996. Each ECG lasts 10 seconds with a sampling rate of 500 Hz. Based on geographic origin, all samples are assumed to be Chagas-negative.

### Usage


To run this project, follow the steps below:

1.**Install dependencies**
 Install the required Python packages using:

```
pip install -r requirements.txt
```

2.**Preprocess the data**

Place the preprocessed PTB-XL and SaMi-Trop datasets (processed using WFDB) into the `original_data/` 

```
original_data/
├── PTB-XL/
│   ├── 01.dat
│   ├── 02.hea
│   ├── ... (additional records)
├── SaMi-Trop/
│   ├── 01.dat
│   ├── 02.hea
│   ├── ... (additional records)
```

Then, run the `preprocessing.py` scripts to generate the `data/` directory.

3.**Train the model**
 Execute the training script:

```
python train.py --data-dir data --use-gpu
```

4.**Model Prediction**

Store the test data files with `.hea` and `.dat` extensions in the `test_data/` directory.

Next, execute the following command. This will generate a `.txt` file containing the prediction probabilities for each sample, which will be saved in the `outputs/` directory.

```
python predict.py \
  --data-dir test_data \
  --model-path model/vHeat_base_data_all_42_80.pth \
  --output-dir outputs
```

5.Evaluate Model

Run the official `evaluate_model.py` script provided by the challenge organizers to obtain the final evaluation scores.

```
python evaluate_model.py \
    --data_folder test_data \
    --output_folder outputs
```

### Result


On our internal test set, we achieved the following performance:

```
Challenge score:0.383
AUROC:0.998
AUPRC:0.993
Accuracy:0.994
F-measure:0.976
```
