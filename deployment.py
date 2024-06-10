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

# Main Streamlit app
def main():
    st.title('Firefighters Clustering Analysis')

    # Load data
    df1 = pd.read_csv("capstone_example.csv")

    # Preprocess and cluster data
    df, cluster = preprocess_and_cluster_data(df1)
    df1['cluster'] = cluster
    df1.drop(['code 1',  'id\'s'], axis=1, inplace=True)
    df1.to_csv("capstone_example_clustered.csv")

    # Visualize clusters in 3D
    st.subheader('Clusters Visualization (3D Scatter Plot)')
    fig = px.scatter_3d(df1, x='leeftijd', y='ervaring (jaren)', z='afstand',
                        color='cluster', symbol='cluster',
                        title='Clustering of Firefighters (3D Visualization)',
                        labels={'cluster': 'Cluster'})
    fig.update_layout(legend=dict(orientation='h', x=1, y=1, xanchor='right', yanchor='top'))
    st.plotly_chart(fig)

    # Generate cluster profiles
    cluster_profiles = generate_cluster_profiles(df1)

    # Display cluster profiles
    st.subheader('Volunteer profiles')

    # Get cluster labels
    cluster_labels = list(cluster_profiles.keys())

    # Create a selection box for cluster selection
    selected_cluster = st.selectbox("Select Cluster", cluster_labels)

    # Display selected cluster profile
    st.write(f"**Cluster {selected_cluster} Profile:**")
    st.write(cluster_profiles[selected_cluster])
    st.write('---')

    # Display detailed information per cluster
    st.subheader('Detailed Information per Clustered volunteer')
    selected_cluster = st.selectbox('Select a cluster to view details:', df1['cluster'].unique())
    cluster_data = df1[df1['cluster'] == selected_cluster]
    st.write(f"**Details for Cluster {selected_cluster}:**")
    st.write(cluster_data)

if __name__ == '__main__':
    main()
