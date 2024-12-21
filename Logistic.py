{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e223c5bd-21da-4e5c-83be-831a6a5b5f63",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import joblib\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import shap\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Load the model\n",
    "model = joblib.load('Logistic.pkl')\n",
    "\n",
    "# Define feature names\n",
    "feature_names = [\"LDH\", \"ALP\", \"Neutrophils\", \"DBIL\", \"ALB\", \"Fg\"]\n",
    "\n",
    "# Streamlit user interface\n",
    "st.title(\"COVID-19 Subphenotype Classifier\")\n",
    "\n",
    "# LDH: numerical input\n",
    "LDH = st.number_input(\"LDH:\", min_value=50, max_value=4000, value=270)\n",
    "\n",
    "# ALP: numerical input\n",
    "ALP = st.number_input(\"ALP:\", min_value=1, max_value=2000, value=80)\n",
    "\n",
    "# Neutrophils: numerical input\n",
    "Neutrophils = st.number_input(\"Neutrophils:\", min_value=0, max_value=50, value=6)\n",
    "\n",
    "# DBIL: numerical input\n",
    "DBIL = st.number_input(\"DBIL:\", min_value=0, max_value=100, value=5)\n",
    "\n",
    "# ALB: numerical input\n",
    "ALB = st.number_input(\"ALB:\", min_value=0, max_value=100, value=35)\n",
    "\n",
    "# Fg: numerical input\n",
    "Fg = st.number_input(\"Fg:\", min_value=0, max_value=50, value=3)\n",
    "\n",
    "# Process inputs and make predictions\n",
    "feature_values = [LDH, ALP, Neutrophils, DBIL, ALB, Fg]\n",
    "features = np.array([feature_values])\n",
    "\n",
    "if st.button(\"Predict\"):\n",
    "    # Predict probabilities\n",
    "    predicted_proba = model.predict_proba(features)[0]\n",
    "    \n",
    "    # 根据预测概率的最高值来确定预测类别（但这里我们直接根据概率阈值判断）  \n",
    "    high_risk_threshold = 0.24  # 24% 的阈值  \n",
    "    if predicted_proba[1] > high_risk_threshold:  # 假设模型输出的第二个概率是高风险类的概率  \n",
    "        predicted_class = 1  # 高风险  \n",
    "    else:  \n",
    "        predicted_class = 0  # 低风险\n",
    "\n",
    "\n",
    "     # 显示预测结果  \n",
    "    st.write(f\"**Predicted Class (Based on Probability Threshold)**: {'Cluster 2' if predicted_class == 1 else 'Cluster 1'}\")  \n",
    "    st.write(f\"**Predicted Probability of Cluster 2**: {predicted_proba[1] * 100:.1f}%\")  "
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
