# Project Title

`Using synthetic data in Child and Adolescent Mental Healthcare – a resource planning feasibility study`

## Description

This project explores the use of synthetic data for resource planning in child and adolescent mental healthcare (CAMH). By leveraging synthetic data, we aim to optimize resource allocation while preserving patient privacy. The study covers data generation, validation, model development, and a feasibility analysis.

## Project Strucutre
```
├── notebooks/
│   ├── exploratory/  
│   │   ├── prepare_real_data.ipynb   # Jupyter notebook for exploring and preparing real data
│   │
│   ├── modelling/  
│   │   ├── synthetise and evaluate quality and privacy/
            ├── synthetise_and_evaluate.ipynb.ipynb # Jupyter notebook for synthetic data modeling and quality/privacy evaluation.
        ├── transform and evaluate utilty/
            ├── prepare_data.ipynb # Jupyter notebook for data preparation for utility evaluation
            ├── classification.ipynb # Train-Synthetic-Test-Real (TSTR) evaluation
            ├── clustering.ipynb # Jupyter notebook for  Different distance-based clustering algorithms for mixed  synthetic/real data.
├── util/
   ├── functions.py   # Helper functions.
├── data/ # available upon request.
│
├── requirements.txt                  # List of Python dependencies required for the project
├── README.md                         # Project overview and instructions (this file)
├── LICENSE                           # License for the project
└── .gitignore                        # Specifies files and directories to be ignored by git
```

## Installation

Follow these steps:

1. **Clone the repo**

`git clone https://github.com/mounirHai/SynthCAMH.git`

`cd SynthCAMH`

2. **Create a Conda environment**

`conda create --name myenv --file requirements.txt`

3. **Activate the Conda Environment**

`conda activate myenv`

4. **Install Dependencies/Requirments**
