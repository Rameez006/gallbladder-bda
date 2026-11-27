# Gallbladder Cancer Detection and Risk Prediction Using Big Data Analytics

A complete college-level Big Data Analytics and Machine Learning mini project demonstrating how Apache Spark (PySpark), Scikit-learn, Plotly, and Streamlit work together on medical tabular datasets.

> **Educational Disclaimer:** The built-in dataset generator creates **synthetic educational records** grounded in published risk-factor associations. It is strictly for academic learning and is **not** a clinical diagnostic tool.

## Project Folder Structure

```
python_project/
├── app.py                  # Module 7: Complete Streamlit Interactive Dashboard
├── requirements.txt        # Python dependencies
├── README.md               # Setup, architecture, and execution guide
├── data/
│   └── sample_gallbladder_dataset.csv
└── src/
    ├── data_loader.py      # Module 1: Synthetic generator & CSV loader
    ├── preprocessing.py    # Module 2: Cleaning, encoding, leak-free splitting
    ├── spark_analytics.py  # Module 3: PySpark distributed aggregations
    ├── eda_visualization.py# Module 4: Plotly, Seaborn, Matplotlib charts
    ├── ml_models.py        # Module 5: LR, DT, RF, SVM training & evaluation
    └── predictor.py        # Module 6: Risk tier inference & disclaimers
```

## Installation & Execution

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies (Java 8/11/17 required for PySpark)
pip install -r requirements.txt

# 3. Launch the Streamlit application
streamlit run app.py
```

## Public Dataset Options

If you wish to replace the synthetic educational dataset with real retrospective clinical data:
1. **SEER Database (NCI)**: Surveillance, Epidemiology, and End Results Program (requires free research registration).
2. **UCI / Kaggle Clinical Datasets**: Search for gallbladder or biliary tract disease cohorts and map columns (`Age`, `Gender`, `Gallstones`, `Wall_Thickness_mm`, `Cancer_Risk`).
