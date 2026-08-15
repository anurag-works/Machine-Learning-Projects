# SmartCart – Customer Segmentation

A customer segmentation project that groups retail customers into distinct segments using unsupervised machine learning (KMeans and Agglomerative Clustering), based on demographic, spending, and purchasing-behavior data.

## Overview

The goal of this project is to segment a retail company's customers into meaningful groups so that marketing and product decisions can be tailored to each segment, rather than treating all customers the same. The pipeline covers data cleaning, feature engineering, dimensionality reduction (PCA), and clustering, along with visual exploration of the results.

## Dataset

The dataset (`smartcart_customers.csv`) contains 2,240 customer records with 22 attributes, including:

- **Demographics**: `Year_Birth`, `Education`, `Marital_Status`, `Income`, `Kidhome`, `Teenhome`, `Dt_Customer`
- **Spending (last 2 years)**: `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds`
- **Purchasing behavior**: `NumDealsPurchases`, `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, `NumWebVisitsMonth`
- **Engagement**: `Recency`, `Complain`, `Response`

Structured similarly to the [Customer Personality Analysis dataset on Kaggle](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis).

## Approach

1. **Data Cleaning**
   - Filled missing `Income` values with the median.
   - Removed outliers (age > 90, income > ₹600,000).

2. **Feature Engineering**
   - `Age` from `Year_Birth`.
   - `Customer_Tenure` — days since enrollment, relative to the most recent signup.
   - `Total_spending` — combined spend across all product categories.
   - `Total_Purchases` — combined purchases across all channels.
   - `Total_Children` — `Kidhome` + `Teenhome`.
   - Grouped `Education` into Undergraduate / Graduate / PostGraduate.
   - Grouped `Marital_Status` into `Living_With` (Alone / Partner).

3. **Exploratory Data Analysis**
   - Pairplots and a correlation heatmap to understand feature relationships and spot outliers.

4. **Encoding & Scaling**
   - Label-encoded categorical features (`Education`, `Living_With`).
   - Standardized continuous features with `StandardScaler`.

5. **Dimensionality Reduction**
   - Applied PCA (3 components) to the scaled continuous features for visualization and clustering.

6. **Choosing k**
   - Used the elbow method (WCSS) and silhouette scores across a range of cluster counts to guide the choice of `k`.

7. **Clustering**
   - Applied both **KMeans** and **Agglomerative Clustering**, and visualized the resulting segments in 3D using the PCA components.

## Tech Stack

- Python
- pandas
- seaborn / matplotlib
- scikit-learn (`StandardScaler`, `PCA`, `KMeans`, `AgglomerativeClustering`, `LabelEncoder`, `silhouette_score`)
- kneed (`KneeLocator`)

## Getting Started

### Prerequisites

```bash
pip install pandas numpy seaborn matplotlib scikit-learn kneed
```

### Running

1. Clone this repository.
2. Place `smartcart_customers.csv` in the project root (or update the path in the notebook).
3. Open and run `Smartcart_clusters.ipynb` in Jupyter Notebook / JupyterLab.

## Project Structure

```
├── Smartcart_clusters.ipynb   # Main analysis notebook
├── smartcart_customers.csv    # Dataset
└── README.md
```

## Future Improvements

- Profile each cluster (e.g., average income, spending, family size) to describe what each segment represents.
- Compare KMeans and Agglomerative results directly (e.g., agreement, silhouette score per method).
- Try additional values of `k` and clustering algorithms (e.g., DBSCAN).

## Acknowledgements

Dataset structure based on the [Customer Personality Analysis dataset](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis) on Kaggle.
