# Early Warning System for Student Churn: A Data-Driven Approach

**Author**: Nisarg Patel  
**Date**: 6th January 2026

## Executive Summary
Built an Early Warning System for an Online University that identifies at-risk students by Week 4 with 70% precision. Discovered that behavioral 'momentum' (Relative Engagement) is a 5x stronger predictor of success than student demographics.

## Key Insights
- **The Week 0 Gap**: Early engagement is critical. Students who engage before the course starts are significantly more likely to succeed.
- **Consistency > Volume**: Regular study habits outweigh cramming. Consistent interaction with the learning management system is a key indicator.
- **Social Connectivity**: Peer interaction is a strong retention signal.

## Technical Stack
- **Data Handling**: Pandas (Chunking for 10M+ rows), NumPy.
- **Machine Learning**: Random Forest Classifier, Scikit-Learn.
- **Visualizations**: Matplotlib/Seaborn.

## Dataset
The dataset used in this project is the Open University Learning Analytics Dataset (OULAD).
You can download it here: [OULAD Dataset](https://analyse.kmi.open.ac.uk/open-dataset)

## Project Structure
- **/data**: Contains the dataset files (OULAD) - Empty for now, download the dataset from the provided link and place it here.
- **/notebooks**: Interactive Jupyter notebook (`Education.ipynb`) containing the analysis and modeling.
- **/visualizations**: Exported charts and graphs (Feature Importance, Week 0 Gap, Confusion Matrix).


## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Navigate to the notebooks directory and launch Jupyter:
   ```bash
   cd notebooks
   jupyter notebook Education.ipynb
   ```
3. Run all cells to reproduce the analysis.
