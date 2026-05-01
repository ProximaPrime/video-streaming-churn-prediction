# 🎬 Customer Churn Prediction — Video Streaming Service

## 👤 Author
Abiodun Adeteye

## 📌 Project Overview
This project builds a machine learning pipeline to predict customer churn for a video streaming platform. The goal is to estimate the probability that a user will cancel their subscription in the next month using behavioral, engagement, and billing-related features.

## 🎯 Objective
Predict churn probability for each customer:
- Target variable: `Churn` (1 = churn, 0 = retained)
- Output: `predicted_probability` (value between 0 and 1)

## 📂 Project Structure

```
video-streaming-churn-prediction/
│
├── train.csv
├── test.csv
├── data_description.csv
├── ChurnPrediction.ipynb
├── prediction_submission.csv
└── README.md
```

## ⚙️ Pipeline Overview
1. Load training and test datasets  
2. Separate features and target (`Churn`)  
3. Handle missing values:
   - Numerical → median  
   - Categorical → "Unknown"  
4. Feature engineering:
   - Engagement = ViewingHoursPerWeek × AverageViewingDuration  
   - CostEfficiency = MonthlyCharges / (ViewingHoursPerWeek + 1)  
   - SupportLoad = SupportTicketsPerMonth / (AccountAge + 1)  
5. Encode categorical variables using target encoding (mean churn rate per category)  
6. Perform stratified train/validation split  
7. Train LightGBM classifier with class imbalance handling (`class_weight="balanced"`)  
8. Evaluate model using ROC AUC score  
9. Retrain final model on full dataset  
10. Predict churn probabilities for test set  
11. Generate submission file  

## 🛠️ Requirements
pip install pandas numpy scikit-learn lightgbm matplotlib

## ▶️ How to Run
1. Clone repository  
git clone <your-repo-link>  
cd <your-project-folder>  

2. Install dependencies  
pip install pandas numpy scikit-learn lightgbm matplotlib  

3. Add dataset files  
Place `train.csv` and `test.csv` in the project directory  

4. Run pipeline  
- Open `ChurnPrediction.ipynb` and run all cells  

5. Output file  
prediction_submission.csv will be generated automatically  

## 📁 Output Format
CustomerID → unique identifier  
predicted_probability → churn likelihood (0–1)

## 📊 Results
- Model: LightGBM Classifier  
- Metric: ROC AUC  
- Score: ~0.75 – 0.80  

Performance highlights:
- Strong ranking ability for churn prediction  
- Stable training without crashes  
- Handles class imbalance effectively  
- Fast execution on large datasets  

## 🚀 Key Features
- Memory-efficient pipeline suitable for 8GB RAM systems  
- Fast training and inference  
- No heavy ensemble models to ensure stability  
- Clean feature engineering for improved signal  
- Production-ready machine learning workflow  

## 🏁 Conclusion
This project delivers a stable and efficient churn prediction system designed for real-world deployment. It prioritizes simplicity, reliability, and performance over overly complex models. The combination of feature engineering and a well-tuned LightGBM model provides strong predictive capability while maintaining low computational cost.
