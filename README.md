# Project Title

`Using synthetic data in Child and Adolescent Mental Healthcare – a resource planning feasibility study`

## Description

This project explores the use of synthetic data for resource planning in child and adolescent mental healthcare (CAMH). By leveraging synthetic data, we aim to optimize resource allocation while preserving patient privacy. The study covers data generation, validation, model development, and a feasibility analysis.

## Project Strucutre

├── notebooks/
│ ├── exploratory/  
│ │ ├── prepare_real_data.ipynb # Jupyter notebook for exploring and transforming/anonymizing data befor synthetising
│ │
│ ├── modelling/  
│ │ ├── synthetise and evaluate quality and privacy/ 
      ├── synthetise_and_evaluate.ipynb.ipynb # Jupyter notebook for synthetic data modeling and quality/privacy evaluation
	├── transform and evaluate utilty/
	  ├── 1_prepare_data.ipynb # Jupyter notebook for data preparation for utility evaluation
	  ├── 2_classification.ipynb # Train-Synthetic-Test-Real (TSTR) evaluation
	  ├── 3_clustering.ipynb # Jupyter notebook for  Different distance-based clustering algorithms for mixed  synthetic/real data.
│
├── requirements.txt # List of Python dependencies
├── README.md # Project overview and instructions
├── LICENSE # License for the project
└── .gitignore # Git ignore file

## Installation

Follow these steps:

1. **Clone the repo**

`git clone https://github.com/mounirHai/SynthCAMH.git`

`cd SynthCAMH`

2. **Create a Conda environment**

`conda create --name myenv python=3.11`

3. **Activate the Conda Environment**

`conda activate myenv`

4. **Install Dependencies/Requirments**
