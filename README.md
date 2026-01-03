
# BatchMind AI  
Explainable & Predictive Intelligence for Zero-Defect Manufacturing

---

## Overview

BatchMind AI is an end-to-end decision support system for manufacturing that predicts batch deviations before they occur, explains the root causes, estimates financial impact, and recommends preventive actions through an interactive dashboard.

---

## Problem Statement

Manufacturing batch deviations lead to scrap, rework, production delays, quality issues, and financial loss. Most existing systems detect problems only after deviation occurs and provide little insight into why it happened or how to prevent it.

BatchMind AI aims to provide early detection with explainability and actionable insights.

---

## Key Features

- Deviation risk prediction using machine learning  
- Risk level classification (LOW, MEDIUM, HIGH)  
- Explainable AI with root cause identification  
- Expected financial loss estimation  
- Severity tagging (MONITOR, ACT SOON, IMMEDIATE ACTION)  
- Cost-based alert system to avoid alert fatigue  
- Rule-based preventive action recommendations  
- Manual and live simulation modes  
- Recent batch history tracking  

---

## System Architecture

User Input → ML Prediction → Explainability → Business Impact Estimation → Recommendations → Dashboard

---

## Dataset

- Synthetic manufacturing batch dataset  
- 1000+ samples  
- Features:
  - Temperature  
  - Pressure  
  - Process duration  
  - Material quality  
  - Machine load  
- Label:
  - Deviation = 1 when parameters cross operational thresholds  
  - Deviation = 0 otherwise  

Synthetic data is used due to confidentiality of real manufacturing data.

---

## Model Training

- Model: Random Forest Classifier  
- Reason for choice:
  - Interpretable
  - Handles non-linear relationships
  - Robust to noise
  - Provides feature importance for explainability  
- Data preprocessing:
  - Feature scaling
  - Train-test split  
- Evaluation metrics:
  - Accuracy
  - Precision
  - Recall  

---

## Risk & Business Logic

- Risk Score: Probability of deviation (0–1)  
- Risk Level:
  - LOW (< 0.30)
  - MEDIUM (0.30–0.60)
  - HIGH (> 0.60)  
- Expected Financial Loss:
  
  Expected Loss = Risk Score × Batch Value × Loss Rate  

- Severity:
  - MONITOR
  - ACT SOON
  - IMMEDIATE ACTION  
- Cost-Based Alert:
  - Triggered when expected loss exceeds ₹50,000  

---

## Explainability

- Feature importance extracted from the trained model  
- Top contributing parameters shown for each prediction  
- Human-readable explanations provided for operator understanding  

---

## Recommendations Engine

- Rule-based corrective actions  
- Examples:
  - High temperature → Reduce temperature
  - Low material quality → Perform quality inspection
  - High machine load → Redistribute load  

---

## Dashboard Components

- Batch parameter simulation (Manual and Live mode)  
- Risk score and risk level cards  
- Estimated financial loss and severity display  
- Cost-based alert banner  
- Root cause explanation section  
- Feature contribution graph  
- Risk trend graph (simulated)  
- Recent batch history table  

---

## Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib  
- Streamlit  

---

## How to Run

1. Install dependencies:
```

pip install -r requirements.txt

```
2. Train the model:
```

python models/train_model.py

```
3. Run the dashboard:
```

streamlit run dashboard/app.py

```

---

## Use Cases

- Manufacturing quality monitoring  
- Early deviation detection  
- Decision support for production engineers  
- Cost-aware operational prioritization  

---

## Conclusion

BatchMind AI combines predictive modeling, explainable AI, and business impact estimation into a single integrated system, making it suitable for real-world manufacturing monitoring and academic evaluation.

