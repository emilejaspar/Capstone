import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Function to preprocess data and perform clustering
def preprocess_and_cluster_data(df):
    # Drop non-relevant columns for clustering
    #df.drop(['code 1', 'code 2', 'beschrijving', 'id\'s'], axis=1, inplace=True)

    # Remove 'm' suffix from 'afstand' column
    df['afstand'] = df['afstand'].str.replace(' m', '')

    # Remove 'm' suffix from 'afstand' column
    df['afstand'] = df['afstand'].str.replace(' m', '')

    # Fill missing values in numerical columns with mean, and categorical columns with mode
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    for col in num_cols:
        df[col] = df[col].fillna(df[col].mean())

    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Convert categorical variables to numerical using one-hot encoding
    df = pd.get_dummies(df, columns=['code 1', 'code 2', 'beschrijving', 'id\'s',
                                     'dagdeel', 'persoonlijkheidstype', 'hobbie', 'geslacht', 'dierenliefhebber',
                                     'opleidingsniveau', 'woonplaats'])
    # Scale the data
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df)

    # Perform clustering
    num_clusters = 4
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_df)

    # Add cluster labels to the original dataframe
    df['cluster'] = clusters

    return df, clusters

# Function to generate cluster profiles
def generate_cluster_profiles(df):
    cluster_profiles = {}
    unique_clusters = sorted(df['cluster'].unique())  # Sort unique cluster labels
    for cluster_label in unique_clusters:
        cluster_data = df[df['cluster'] == cluster_label].drop('cluster', axis=1)
        cluster_profile = {}
        for col in cluster_data.columns:
            if cluster_data[col].dtype == 'object':
                mode_value = cluster_data[col].mode().iloc[0]
                cluster_profile[col] = mode_value
            else:
                cluster_profile[col] = round(cluster_data[col].mean(), 0)  # Round numerical values to 2 decimal places
        cluster_profiles[cluster_label] = cluster_profile
    return cluster_profiles

# Load data
df1 = pd.read_csv("capstone_example.csv")

# Preprocess and cluster data
df, cluster = preprocess_and_cluster_data(df1)
df1['cluster'] = cluster
df1.drop(['code 1',  'id\'s'], axis=1, inplace=True)
df1.to_csv("capstone_example_clustered.csv", index=False)

# Generate cluster profiles
cluster_profiles = generate_cluster_profiles(df1)
df_cluster_profiles = pd.DataFrame.from_dict(cluster_profiles)
# Transpose the DataFrame
transposed_df = df_cluster_profiles.transpose()
print(df_cluster_profiles)
transposed_df.to_csv('cluster_profiles.csv')

