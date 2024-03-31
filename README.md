# Predictive Maintenance with Supervised Learning

## Overview

Machines play a critical role in modern industries, and ensuring their optimal performance is essential for business success. Traditional maintenance strategies such as corrective and preventive maintenance have limitations in addressing unexpected failures efficiently. Predictive maintenance offers a proactive approach by leveraging machine learning techniques to monitor and predict the health status of machines, thereby minimizing downtime and reducing costs.

This project focuses on predictive maintenance using supervised learning techniques, specifically binary and multi-class classification. The primary objectives are to predict machine failures and determine the type of failure using Python and its machine learning libraries. By proposing machine learning solutions for predictive maintenance, this project aims to contribute to the Industry 4.0 paradigm, enabling industries to enhance operational efficiency and reduce maintenance costs.

## Problem Statement

The main challenge addressed in this project is to develop accurate models for predictive maintenance. There are two key tasks:

1. Type of Failure Detection: The model should classify the type of failure based on input features.
2. Failure Prediction: The model should predict whether a failure will occur within a given timeframe.

## Dataset

The AI4I 2020 Predictive Maintenance Dataset from the UCI Machine Learning Repository is used for this project. This dataset contains features such as air temperature, process temperature, rotational speed, torque, tool wear, and machine failure indicators. The dataset allows for supervised learning tasks, making it suitable for predictive maintenance analysis.

## Data Source
[AI4I 2020 Predictive Maintenance Dataset](https://doi.org/10.24432/C5HS5C)

## Project Structure

The project follows a structured organization of files and directories:

### predictive_maintenance/

### data/
- ai4i2020.csv

### models/
- machine_failure_prediction.py
- failure_type_detection.py

### notebooks/
- exploratory_data_analysis.ipynb

- README.md
- project_plan.md

### docs/
- project_plan.md
- project_report.md

### scripts/
    - data_preprocessing.py
    - model_training.py
    - model_evaluation.py


## Installation and Dependencies

To run the project, ensure you have Python installed along with the following dependencies:

- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- jupyterlab (for running Jupyter notebooks)

You can install these dependencies using pip:
`pip install numpy pandas scikit-learn matplotlib seaborn jupyterlab`

## Usage

1. Clone the repository:
`git clone https://github.com/Oluwaseun25/.git`

2. Navigate to the project directory:
`cd predictive-maintenance-supervised-learning`

3. Run Jupyter notebooks for data analysis and model development:
`jupyter lab`

4. Execute Python scripts for machine learning models:
`python models/machine_failure_prediction.py`
`python models/failure_type_detection.py`

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## REFRENCE
`AI4I 2020 Predictive Maintenance Dataset. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C.`