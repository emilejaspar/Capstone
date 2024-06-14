# Main Streamlit app
import streamlit as st
import pandas as pd
import plotly.express as px

# Function to get detailed column information
def get_column_info(df, column):
    col_info = {}
    if df[column].dtype == 'object':
        col_info['value_counts'] = df[column].value_counts()
    else:
        col_info['median'] = df[column].median()
        col_info['std'] = round(df[column].std(),0)
    return col_info

def main():
    # Set wider page configuration
    st.set_page_config(layout="wide")

    st.title('Firefighters Clustering Analysis')

    # Load data
    df1 = pd.read_csv("capstone_example_clustered.csv")
    df_cluster_profiles = pd.read_csv("cluster_profiles.csv", index_col=0)
    #cluster_profiles = df_cluster_profiles.to_dict(orient='list')

    # Visualize clusters in 3D
    height = 600  # Set your desired height here

    st.subheader('Clusters Visualization (3D Scatter Plot)')
    fig = px.scatter_3d(df1, x='leeftijd', y='ervaring (jaren)', z='afstand',
                        color='cluster', symbol='cluster',
                        title='Clustering of Firefighters (3D Visualization)',
                        labels={'cluster': 'Cluster'}, height=height)
    fig.update_layout(legend=dict(orientation='h', x=1, y=1, xanchor='right', yanchor='top'))
    st.plotly_chart(fig)

    # Display cluster profiles
    st.subheader('Volunteer profiles')

    # Get cluster labels (using the first column as index)
    cluster_labels = df_cluster_profiles.index.tolist()

    # Create a selection box for cluster selection
    selected_cluster_index = st.selectbox("Select Cluster", range(len(cluster_labels)))

    # Get the selected cluster label
    selected_cluster_label = cluster_labels[selected_cluster_index]

    # Get the cluster_profile
    cluster_profile = df_cluster_profiles.loc[selected_cluster_label]

    # Filter the DataFrame based on similarity to cluster_profile
    similar_rows = df_cluster_profiles[df_cluster_profiles.index == selected_cluster_label]

    # Display the filtered row
    st.write(similar_rows)
    st.write('---')

    # Display detailed information per cluster
    st.subheader('Detailed Information per Cluster per Column')
    unique_clusters = sorted(df1['cluster'].unique())
    selected_cluster = st.selectbox('Select a cluster to view details:', unique_clusters)
    cluster_data = df1[df1['cluster'] == selected_cluster]
    selected_column = st.selectbox('Select a column to view details:', cluster_data.columns)

    if selected_column:
        column_info = get_column_info(cluster_data, selected_column)
        st.write(f"**Detailed information for column: {selected_column} in Cluster {selected_cluster}**")
        if 'value_counts' in column_info:
            st.write(column_info['value_counts'])
        else:
            st.write(f"Median: {column_info['median']}")
            st.write(f"Standard Deviation: {column_info['std']}")


if __name__ == '__main__':
    main()