# Main Streamlit app
import streamlit as st
import pandas as pd
import plotly.express as px

def main():
    st.title('Firefighters Clustering Analysis')

    # Load data
    df1 = pd.read_csv("capstone_example_clustered.csv")
    df_cluster_profiles = pd.read_csv("cluster_profiles.csv", index_col=0)
    cluster_profiles = df_cluster_profiles.to_dict(orient='list')

    # Define color mapping for clusters with softer colors
    color_map_soft = {
        0: 'rgb(255, 102, 102)',  # Soft red
        1: 'rgb(102, 178, 255)',  # Soft blue
        2: 'rgb(152, 255, 152)',  # Soft green
        3: 'rgb(255, 230, 179)',  # Soft yellow
        # Add more colors for additional clusters if needed
    }

    # Map cluster labels to softer colors
    df1['color'] = df1['cluster'].map(color_map_soft)

    # Visualize clusters in 3D
    st.subheader('Clusters Visualization (3D Scatter Plot)')
    fig = px.scatter_3d(df1, x='leeftijd', y='ervaring (jaren)', z='afstand',
                        color='color', symbol='cluster',
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