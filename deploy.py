# Main Streamlit app
import streamlit as st
import pandas as pd
import plotly.express as px

def main():
    st.title('Firefighters Clustering Analysis')

    # Load data
    df1 = pd.read_csv("capstone_example_clustered.csv")

    # Visualize clusters in 3D
    st.subheader('Clusters Visualization (3D Scatter Plot)')
    fig = px.scatter_3d(df1, x='leeftijd', y='ervaring (jaren)', z='afstand',
                        color='cluster', symbol='cluster',
                        title='Clustering of Firefighters (3D Visualization)',
                        labels={'cluster': 'Cluster'})
    fig.update_layout(legend=dict(orientation='h', x=1, y=1, xanchor='right', yanchor='top'))
    st.plotly_chart(fig)

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