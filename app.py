import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.title("Customer Segmentation App")

# Sample Data
data = {
    'Age': [25, 45, 35, 23, 52, 40, 60, 48],
    'Income': [20000, 80000, 50000, 22000, 90000, 62000, 100000, 75000],
    'Spending': [30, 80, 60, 20, 90, 70, 95, 85]
}

df = pd.DataFrame(data)

st.subheader("Customer Data")
st.write(df)

# Select number of clusters
k = st.slider("Select number of clusters", 2, 5, 3)

# Clustering
X = df[['Income', 'Spending']]
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

st.subheader("Clustered Data")
st.write(df)

# Plot
fig, ax = plt.subplots()
ax.scatter(df['Income'], df['Spending'], c=df['Cluster'], cmap='viridis')
ax.set_xlabel('Income')
ax.set_ylabel('Spending')
ax.set_title('Customer Segments')

st.pyplot(fig)