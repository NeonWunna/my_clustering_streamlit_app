#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on Sun Apr 20 15:51:29 2025

@author: thutazaw
"""

import streamlit as st 
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

#load model 
with open ('kmeans_model.pkl', 'rb')as f:
    loaded_model = pickle.load(f)
    
#set the page config
st.set_page_config(page_title="k-Means Clustering App", layout="centered")

#set title
st.title("k-Means Clustering Visualizer by Wunna Kaung Myat")

#Display cluster centers
# =============================================================================
# st.subheader("Example Data for Visualization")
# st.markdown("This demo uses example data (2D) to illustrate clustering results.")
# =============================================================================

#Load from a saved dataset or generate synthetic data
X, _ = make_blobs(n_samples=300, centers=loaded_model.n_clusters, cluster_std=0.60, random_state=0)

# Predict using the loaded model
y_kmeans = loaded_model.predict(X)

#plotting
fig, ax = plt.subplots()
scatter = ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
ax.scatter(loaded_model.cluster_centers_[:, 0], loaded_model.cluster_centers_[:, 1], s=300, c='red')
ax.set_title('k-Means Clustering')
ax.legend()
st.pyplot(fig)