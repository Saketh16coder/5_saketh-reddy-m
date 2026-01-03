
# BatchMind AI  
Explainable, Predictive & Generative Intelligence for Zero-Defect Manufacturing

---

## Overview

BatchMind AI is an end-to-end decision support system for manufacturing that predicts batch deviations before they occur, explains root causes, estimates financial impact, recommends preventive actions, and generates human-readable manufacturing insights using Generative AI through an interactive dashboard.

---

## Problem Statement

Manufacturing batch deviations lead to scrap, rework, production delays, quality issues, and financial loss. Most existing systems detect problems only after deviation occurs and provide limited insight into why it happened or how to prevent it.

BatchMind AI focuses on early deviation prediction with explainability, business impact awareness, and AI-assisted decision support.

---

## Key Features

- Deviation risk prediction using machine learning  
- Risk level classification (LOW, MEDIUM, HIGH)  
- Explainable AI with root cause identification  
- Expected financial loss estimation  
- Severity tagging (MONITOR, ACT SOON, IMMEDIATE ACTION)  
- Cost-based alert system to avoid alert fatigue  
- Rule-based preventive action recommendations  
- Generative AI–based manufacturing insights  
- AI risk narrative and what-if reasoning  
- AI confidence and trust disclaimer  
- Manual and live simulation modes  
- Recent batch history tracking  

---

## System Architecture

User Input  
→ ML Prediction  
→ Explainability  
→ Business Impact Estimation  
→ Recommendations  
→ Generative AI Insight Layer  
→ Dashboard

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
- Interpretable and robust to non-linear relationships  
- Feature importance used for explainability  
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

- Feature importance derived from the trained model  
- Top contributing parameters highlighted per prediction  
- Human-readable explanations for operator understanding  

---

## Recommendations Engine

- Rule-based corrective action system  
- Examples:
  - High temperature → Reduce operating temperature  
  - Low material quality → Perform quality inspection  
  - High machine load → Redistribute machine load  

---

## Generative AI Insights

- Converts ML predictions and explainability outputs into structured insights  
- Generates AI risk narrative, what-if reasoning, and confidence note  
- Designed as a decision-support layer  
- Does not replace prediction or engineering judgment  
- Graceful fallback ensures insight availability without external LLM access  

---

## Dashboard Components

- Batch parameter simulation (Manual and Live mode)  
- Risk score and risk level cards  
- Estimated financial loss and severity display  
- Cost-based alert banner  
- Root cause explanation section  
- Feature contribution visualization  
- Risk trend visualization  
- Generative AI manufacturing insights panel  
- Recent batch history table  

---

## Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Streamlit  

---

## How to Run

1. Install dependencies:
```

pip install -r requirements.txt

```


## Generative AI Setup

BatchMind AI uses Generative AI to convert predictive and explainable model outputs into human-readable manufacturing insights.

### Set API Key

Windows:
```

setx OPENAI_API_KEY "your_api_key_here"

```

Linux / macOS:
```

export OPENAI_API_KEY="your_api_key_here"

```

Restart the terminal before running the dashboard.

If the API key is not set or external access is unavailable, the system automatically falls back to a deterministic AI insight generator to ensure uninterrupted functionality.



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
- AI-assisted decision support  
- Cost-aware operational prioritization  

---

## Conclusion

BatchMind AI integrates predictive machine learning, explainable AI, business impact analysis, and Generative AI–based insight generation into a single system for proactive manufacturing deviation prevention.

