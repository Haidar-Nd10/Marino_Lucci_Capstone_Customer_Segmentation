# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 19:41:10 2023

@author: Haidar
"""

import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder  
import random
from sklearn.preprocessing import StandardScaler
import hydralit_components as hc
from apyori import apriori
#import openai

# Set your OpenAI API key
#openai.api_key = "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"


# Set page configuration
st.set_page_config(page_title="Customer Segmentation by Demographics", layout="wide")

# Load the dataset
data = pd.read_csv('Data_full_with_Gender.csv')

# Clean and preprocess data
# Create a dictionary to hold the most frequent locations for each category
category_frequent_locations = data.groupby('Client Category')['Location'].apply(lambda x: x.mode().iloc[0]).to_dict()

# Impute missing 'Location' values based on Client Category
for category, frequent_location in category_frequent_locations.items():
    data.loc[(data['Client Category'] == category) & (data['Location'].isnull()), 'Location'] = frequent_location

# Create a dictionary to hold the mean ages for each category
category_mean_ages = data.groupby('Client Category')['Client_Age'].mean().to_dict()

# Impute missing 'Client_Age' values based on Client Category
for category, mean_age in category_mean_ages.items():
    data.loc[(data['Client Category'] == category) & (data['Client_Age'].isnull()), 'Client_Age'] = mean_age

# Convert the 'Client_Age' column to integer data type
data['Client_Age'] = data['Client_Age'].astype(int)


#############Page 1: Demographics

# Create a Streamlit app
def Demographics_Segmentation():
    st.title("Customer Segmentation by Demographics")
    
    ## Market Basket Engine
    st.markdown("## (1) Apply Elbow Method for Segmentation")
    
    # Add a small title to the sidebar
    st.sidebar.markdown("**(1)Segmentation Tools:**")
    
    # Sidebar with dropdown to select Client Category
    selected_category = st.sidebar.selectbox("Select Client Category", data['Client Category'].unique())

    # Subset data based on selected category
    demo_subset = data[data['Client Category'] == selected_category][['Client_Type', 'Location', 'Client_Age', 'Gender']]

    # Apply One-Hot Encoding to categorical columns
    demo_subset_encoded = pd.get_dummies(demo_subset, columns=['Client_Type', 'Location', 'Gender'], drop_first=True)

    # Elbow Method
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(demo_subset_encoded)
        sse.append(kmeans.inertia_)

    # Display Elbow Method plot
    with st.container():
        plt.figure(figsize=(6, 5))
        plt.plot(range(1, 11), sse, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('SSE (Sum of Squared Errors)')
        plt.title('Elbow Method')
        st.pyplot(plt)


    # Slider to choose the number of clusters
    n_clusters = st.sidebar.slider("Choose Number of Clusters using elbow method", 1, 20, 1)

    data_cluster = data[data['Client Category'] == selected_category]

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data_cluster['Cluster'] = kmeans.fit_predict(demo_subset_encoded)

    # Show table of clustered data
    st.write("Resulting Clustered Data:")
    st.dataframe(data_cluster)

    # Select relevant features for analysis
    numeric_feature = ['Client_Age', 'Cluster']
    
    # Select categorical features for analysis
    categorical_features = ['Client_Type', 'Location', 'Gender']

    # Get a list of random colors for clusters
    cluster_colors = [get_random_color() for _ in range(n_clusters)]
    
   # Display a header above the DataFrame with smaller font size
    st.markdown("<h3 style='font-size: 18px;'>Visuals to define Clusters:</h3>", unsafe_allow_html=True)
    
   # Arrange buttons in a horizontal layout
    col1, col2, col3, col4 = st.columns(4)
    
    # Button to plot Cluster vs Client_Type
    if col1.button("Cluster vs Client_Type"):
        plot_cluster_vs_client_type(data_cluster, cluster_colors)
    
    # Button to plot Cluster vs Location
    if col2.button("Cluster vs Location"):
        plot_cluster_vs_location(data_cluster, cluster_colors)
    
    # Button to plot Cluster vs Gender
    if col3.button("Cluster vs Gender"):
        plot_cluster_vs_gender(data_cluster, cluster_colors)
    
    # Button to plot Cluster vs Client_Age
    if col4.button("Cluster vs Client_Age"):
        plot_cluster_vs_client_age(data_cluster, cluster_colors)
        
    
    # Display a header above the DataFrame with smaller font size
    st.markdown("<h3 style='font-size: 18px;'>Search Client Cluster:</h3>", unsafe_allow_html=True)
    
     # Multiselectbox to search for Client name and display Cluster
    selected_clients = st.multiselect("Search and Select Client Names", data_cluster['Client'].unique())
    
    # Display selected clients and their clusters
    selected_clients_clusters = {}  # Dictionary to store selected clients and their clusters
    
    if selected_clients:
        for client in selected_clients:
            client_cluster = data_cluster[data_cluster['Client'] == client]['Cluster'].iloc[0]
            selected_clients_clusters[client] = client_cluster
    
        selected_clients_df = pd.DataFrame(list(selected_clients_clusters.items()), columns=['Client', 'Cluster'])
        st.write("Selected Clients and Their Clusters:")
        st.dataframe(selected_clients_df)

    # Recommendation Engine Section
    st.markdown("## (2) Recommendation Engine")
    
    # Arrange the input elements next to each other using columns
    col1, col2, col3 = st.columns(3)
    
    # Dropdowns for selecting Kind/Cut and Cluster
    with col1:
        selected_groupby_column = st.selectbox("Select Grouping Column (Kind or Cut)", ['Kind', 'Cut'])
    
    with col2:
        selected_cluster_for_recommendation = st.selectbox("Select Cluster for Recommendation", data_cluster['Cluster'].unique())
    
    with col3:
        top_n_items = st.slider("Select Number of Items to Display", 1, 20, 5)

    # Filter data for the selected cluster
    cluster_data = data_cluster[data_cluster['Cluster'] == selected_cluster_for_recommendation]
    
    # Group data by selected column and calculate sum of 'Net Amount'
    grouped_data = cluster_data.groupby(selected_groupby_column)['Net Amount'].sum().reset_index()
    
    # Get top n items based on Net Amount sum
    top_n_items_df = grouped_data.nlargest(top_n_items, 'Net Amount')
    
    # Display the recommendation results in a DataFrame
    st.write("Top", top_n_items, f"{selected_groupby_column} Recommended for Cluster {selected_cluster_for_recommendation} based on interest:")
    st.dataframe(top_n_items_df)
    
    
    ###Customized plot
    # Add a small title to the sidebar
    st.sidebar.markdown("**(2)Create Additional Plot:**")
    # Sidebar with dropdowns to select columns and cluster
    selected_x_column = st.sidebar.selectbox("Select X-axis Column", data_cluster.columns)
    selected_y_column = st.sidebar.selectbox("Select Y-axis Column", data_cluster.columns)
    selected_cluster = st.sidebar.selectbox("Select Cluster", data_cluster['Cluster'].unique())
    selected_plot_type = st.sidebar.selectbox("Select Plot Type", ["Bar", "Line", "Scatter"])

    # Display a header above the DataFrame with smaller font size
    st.markdown("<h3 style='font-size: 18px;'>Custom Visual:</h3>", unsafe_allow_html=True)
     
    # Customized Plot based on selected plot type
    if st.button("Show Additional Created Plot"):
        with st.container():
            plt.figure(figsize=(8, 6))
            scatter_data = data_cluster[data_cluster['Cluster'] == selected_cluster]
        
            if selected_plot_type == "Bar":
                if scatter_data[selected_y_column].dtype == 'float64' and scatter_data[selected_x_column].dtype == 'O':
                    y_values = scatter_data.groupby(selected_x_column)[selected_y_column].sum()  # Calculate sum instead of mean
                    x_values = y_values.index
                    plt.bar(x_values, y_values, color=cluster_colors[selected_cluster])
                    plt.xticks(rotation=90)
                else:
                    plt.bar(scatter_data[selected_x_column], scatter_data[selected_y_column], color=cluster_colors[selected_cluster])
            elif selected_plot_type == "Line":
                if scatter_data[selected_y_column].dtype == 'float64' and scatter_data[selected_x_column].dtype == 'O':
                    y_values = scatter_data.groupby(selected_x_column)[selected_y_column].sum()  # Calculate sum instead of mean
                    x_values = y_values.index
                    plt.plot(x_values, y_values, marker='o', color=cluster_colors[selected_cluster])
                    plt.xticks(rotation=90)
                else:
                    plt.plot(scatter_data[selected_x_column], scatter_data[selected_y_column], marker='o', color=cluster_colors[selected_cluster])
            elif selected_plot_type == "Scatter":
                plt.scatter(scatter_data[selected_x_column], scatter_data[selected_y_column], color=cluster_colors[selected_cluster], label=f"Cluster {selected_cluster}")
        
            plt.xlabel(selected_x_column)
            plt.ylabel(selected_y_column)
            plt.title(f'Sum {selected_y_column} vs {selected_x_column} for Cluster {selected_cluster}')
            plt.legend()
            st.pyplot(plt)
    #With this code, users can now select the type of plot (bar, line, or scatter) they want to see for the comparison between the selected x and y columns. Just make sure to adjust the code to fit your specific data and use case.
     
    ## Market Basket Engine
    st.markdown("## (3) Market Basket Engine")
    # Filter sales data
    data_sales = data_cluster[data_cluster['Invoice Type'] == 'Sales']
    
    # Display a header above the DataFrame with smaller font size
    st.markdown("<h3 style='font-size: 18px;'>Select Market Basket Rules:</h3>", unsafe_allow_html=True)
    
    # Arrange the input elements next to each other using columns
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Filter by Cluster
    with col1:
        selected_cluster_market = st.selectbox("Select Cluster", data_cluster['Cluster'].unique(), key="cluster_select")
    data_selected = data_sales[data_sales['Cluster'] == selected_cluster_market]
    
    # Relevant product columns
    with col2:
        product_columns = ['Kind', 'Cut', 'Product Code', 'Product Name', 'Brand', 'Color', 'Size']
        selected_product_columns = st.selectbox('Select Product Feature', product_columns, key="product_feature_select")
    
    # Input thresholds
    with col3:
        min_support_text = st.text_input('Minimum Support', '0.010', key="min_support_input")
        min_support = float(min_support_text)
    
    with col4:
        min_confidence = st.slider('Minimum Confidence', 0.0, 1.0, 0.05, key="min_confidence_slider")
    with col5:
        min_lift = st.slider('Minimum Lift', 1.0, 10.0, 2.0, key="min_lift_slider")
        min_length = 2  # Always fixed to 2 in your case
    
    # Group transactions based on selected product columns
    transaction = data_selected.groupby(['Client', 'Date'])[selected_product_columns].apply(lambda x: frozenset(x.dropna().values.flatten()))
    
    # Apply Apriori algorithm
    rules = apriori(transaction.tolist(), min_support=min_support, min_confidence=min_confidence, min_lift=min_lift, min_length=min_length)
    results = list(rules)
    
    # Create DataFrame from inspection
    def inspect(results):
        antecedent = [tuple(result[2][0][0])[0] for result in results]
        consequent = [tuple(result[2][0][1])[0] for result in results]
        supports = [result[1] for result in results]
        confidences = [result[2][0][2] for result in results]
        lifts = [result[2][0][3] for result in results]
        return list(zip(antecedent, consequent, supports, confidences, lifts))
    
    basket_df = pd.DataFrame(inspect(results), columns=['Antecedent', 'Consequent', 'Support', 'Confidence', 'Lift'])
    
    # Display a header above the DataFrame with smaller font size
    st.write("Basket Items based on selected rules:</h3>", unsafe_allow_html=True)
    
    # Display the resulting DataFrame
    st.write(basket_df)
    

    
def get_random_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))


# Define a function to plot cluster vs client_type
def plot_cluster_vs_client_type(data_cluster, cluster_colors):
    plt.figure(figsize=(10, 6))
    
    num_clusters = len(data_cluster['Cluster'].unique())
    unique_client_types = data_cluster['Client_Type'].unique()
    width = 0.8 / num_clusters

    for idx, cluster in enumerate(data_cluster['Cluster'].unique()):
        cluster_data = data_cluster[data_cluster['Cluster'] == cluster]
        cluster_counts = cluster_data['Client_Type'].value_counts().reindex(unique_client_types, fill_value=0)
        x = [i + width * idx for i in range(len(unique_client_types))]
        plt.bar(x, cluster_counts, width=width, color=cluster_colors[idx], label=f"Cluster {cluster}")

    plt.title('Cluster vs Client_Type')
    plt.xlabel('Client_Type')
    plt.ylabel('Count')
    plt.legend()
    plt.xticks([i + width * (num_clusters / 2 - 0.5) for i in range(len(unique_client_types))], unique_client_types, rotation=45)
    plt.tight_layout()
    st.pyplot(plt)


# Define a function to plot cluster vs location
def plot_cluster_vs_location(data_cluster, cluster_colors):
    plt.figure(figsize=(10, 6))

    locations = data_cluster['Location'].unique()
    num_clusters = len(data_cluster['Cluster'].unique())
    width = 0.8 / num_clusters

    for idx, cluster in enumerate(data_cluster['Cluster'].unique()):
        cluster_data = data_cluster[data_cluster['Cluster'] == cluster]
        cluster_counts = cluster_data['Location'].value_counts().reindex(locations, fill_value=0)
        x = [i + width * idx for i in range(len(locations))]
        plt.bar(x, cluster_counts, width=width, color=cluster_colors[idx], label=f"Cluster {cluster}")

    plt.title('Cluster vs Location')
    plt.xlabel('Location')
    plt.ylabel('Count')
    plt.legend()
    plt.xticks([i + width * (num_clusters / 2 - 0.5) for i in range(len(locations))], locations, rotation=90)
    plt.tight_layout()
    st.pyplot(plt)

# Define a function to plot cluster vs gender
def plot_cluster_vs_gender(data_cluster, cluster_colors):
    plt.figure(figsize=(10, 6))
    cluster_counts = []

    for cluster in data_cluster['Cluster'].unique():
        cluster_data = data_cluster[data_cluster['Cluster'] == cluster]
        cluster_count = cluster_data['Gender'].value_counts().reindex(cluster_data['Gender'].unique(), fill_value=0)
        cluster_counts.append(cluster_count)

    width = 0.2
    x = range(len(data_cluster['Gender'].unique()))

    for idx, cluster_count in enumerate(cluster_counts):
        plt.bar([pos + width * idx for pos in x], cluster_count, width=width,
                color=cluster_colors[idx], label=f"Cluster {idx}")

    plt.title('Cluster vs Gender')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.legend()
    plt.xticks([pos + width * (len(cluster_counts) / 2 - 0.5) for pos in x], data_cluster['Gender'].unique(), rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

# Define a function to plot cluster vs Client_Age
def plot_cluster_vs_client_age(data_cluster, cluster_colors):
    plt.figure(figsize=(10, 6))
    cluster_data = [data_cluster[data_cluster['Cluster'] == cluster]['Client_Age'] for cluster in data_cluster['Cluster'].unique()]
    
    plt.boxplot(cluster_data, labels=data_cluster['Cluster'].unique(), patch_artist=True)
    
    plt.title('Cluster vs Client_Age')
    plt.xlabel('Cluster')
    plt.ylabel('Client_Age')
    st.pyplot(plt)






############# Page 2: Purchasing Behavior

# Select relevant features for segmentation
selected_columns = ['Net Amount', 'Discount_percent', 'Qty', 'Product_Price']


# Create a Streamlit app for Segmentation by Purchasing Behavior
def Purchasing_Behavior():
    st.title("Customer Segmentation by Purchasing Behavior")
    
    
    st.markdown("## (1) Apply Elbow Method for Segmentation")
    
    # Add a small title to the sidebar
    st.sidebar.markdown("**(1)Segmentation Tools:**")

    # Sidebar with dropdown to select Client Category
    selected_category = st.sidebar.selectbox("Select Client Category", data['Client Category'].unique())
    
    # Apply filtering based on selected Client Category
    filtered_data = data[data['Client Category'] == selected_category]
    
    # Aggregate filtered data
    aggregate_data = filtered_data.groupby('Client').agg({
        'Net Amount': 'sum',
        'Qty': 'sum',
        'Discount_percent': 'mean',
        'Product_Price': 'mean'
    }).reset_index()
    
    
    # If the selected category is 'Retail', remove rows with infinite and NaN values
    if selected_category == 'Retail':
        aggregate_data = aggregate_data[~aggregate_data['Discount_percent'].isin([float('inf'), float('-inf')])]
        aggregate_data = aggregate_data.dropna(subset=['Discount_percent'])

    # Apply standardization to numerical columns
    num_columns = ['Net Amount', 'Qty', 'Product_Price', 'Discount_percent']
    aggregate_data['Discount_percent'] = aggregate_data['Discount_percent'].round(6)
    scaler = StandardScaler()
    scaled_numerical_data = scaler.fit_transform(aggregate_data[num_columns])
    scaled_numerical_df = pd.DataFrame(scaled_numerical_data, columns=num_columns)

    # Elbow Method
    sse = []
    num_clusters_range = range(1, 11)
    for num_clusters in num_clusters_range:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(scaled_numerical_df)
        sse.append(kmeans.inertia_)
    
    # Display Elbow Method plot
    plt.figure(figsize=(10, 6))
    plt.plot(num_clusters_range, sse, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Sum of Squared Distances')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.xticks(num_clusters_range)
    st.pyplot(plt)

    # Slider to choose the number of clusters
    n_clusters = st.sidebar.slider("Choose Number of Clusters", 1, 10, 4)
    
    data_cluster = data[data['Client Category'] == selected_category]

    # If the selected category is 'Retail', remove rows with infinite and NaN values
    if selected_category == 'Retail':
        data_cluster = data_cluster[~data_cluster['Discount_percent'].isin([float('inf'), float('-inf')])]
        data_cluster = data_cluster.dropna(subset=['Discount_percent'])

    
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    aggregate_data['Cluster'] = kmeans.fit_predict(scaled_numerical_df)

    # Show clustered data
    st.write("Resulting Clustered Data:")
    st.dataframe(aggregate_data)
    
    # Create a dictionary mapping client names to their clusters from aggregate_data
    client_to_cluster = dict(zip(aggregate_data['Client'], aggregate_data['Cluster']))

    # Add 'Cluster' column to data_cluster based on the mapping
    data_cluster['Cluster'] = data_cluster['Client'].map(client_to_cluster)
    
    # Drop rows with NaN values in the 'Cluster' column
    data_cluster = data_cluster.dropna(subset=['Cluster'])

    # Change the data type of 'Cluster' column to integer
    data_cluster['Cluster'] = data_cluster['Cluster'].astype(int)
    
    # Define cluster colors
    cluster_colors = ['red', 'green', 'orange', 'blue', 'yellow', 'purple', 'black', 'brown', 'pink', 'grey']

    # Display a header above the DataFrame with smaller font size
    st.markdown("<h3 style='font-size: 18px;'>Visuals to define Clusters:</h3>", unsafe_allow_html=True)
    
   # Button to show pair plot
    if st.button("Show Visual"):
        plot_columns = ['Net Amount', 'Qty', 'Product_Price', 'Discount_percent']
    
        # Create a scatter plot matrix
        fig, axes = plt.subplots(nrows=len(plot_columns), ncols=len(plot_columns), figsize=(12, 12))
    
        cluster_ids = aggregate_data['Cluster'].unique()  # Get available cluster IDs
    
        for i, column1 in enumerate(plot_columns):
            for j, column2 in enumerate(plot_columns):
                ax = axes[i, j]
    
                if i == j:
                    # Histogram for diagonal plots
                    ax.hist(aggregate_data[column1], bins=20, color='skyblue', alpha=0.7)
                else:
                    for cluster_id in cluster_ids:  # Iterate through available clusters
                        cluster_data = aggregate_data[aggregate_data['Cluster'] == cluster_id]
                        scatter = ax.scatter(cluster_data[column2], cluster_data[column1], c=cluster_colors[cluster_id], alpha=0.5)
    
                    # Set axis labels
                    if i == len(plot_columns) - 1:
                        ax.set_xlabel(column2)
                    if j == 0:
                        ax.set_ylabel(column1)
    
                    # Hide axis ticks for better appearance
                    ax.set_xticks([])
                    ax.set_yticks([])
    
        plt.suptitle('Pair Plot of Features by Cluster')
        plt.tight_layout(rect=[0, 0.03, 0.95, 0.9])  # Adjust the rectangle for space
    
        # Show the pair plot
        st.pyplot(plt)
    
        # Create a legend-like plot below the pair plot
        legend_fig, legend_ax = plt.subplots(figsize=(8, 1))
        for cluster_id in sorted(cluster_ids):  # Sort cluster IDs in order
            legend_ax.scatter([], [], c=cluster_colors[cluster_id], label=f'Cluster {cluster_id}')
        legend_ax.set_xlim(0, 1)
        legend_ax.set_ylim(0, 1)
        legend_ax.set_axis_off()
        legend_ax.legend(loc='center', ncol=len(cluster_ids))
        
        # Show the legend-like plot
        st.pyplot(legend_fig)

    
    ##Client Detail:
    # Display a header above the DataFrame with smaller font size
    st.markdown("<h3 style='font-size: 18px;'>Search Client Cluster:</h3>", unsafe_allow_html=True)
    
     # Multiselectbox to search for Client name and display Cluster
    selected_clients = st.multiselect("Search and Select Client Names", data_cluster['Client'].unique())
    
    # Display selected clients and their clusters
    selected_clients_clusters = {}  # Dictionary to store selected clients and their clusters
    
    if selected_clients:
        for client in selected_clients:
            client_cluster = data_cluster[data_cluster['Client'] == client]['Cluster'].iloc[0]
            selected_clients_clusters[client] = client_cluster
    
        selected_clients_df = pd.DataFrame(list(selected_clients_clusters.items()), columns=['Client', 'Cluster'])
        st.write("Selected Clients and Their Clusters:")
        st.dataframe(selected_clients_df)
    

    
    # Recommendation Engine Section
    st.markdown("## (2) Recommendation Engine")
    
    # Arrange the input elements next to each other using columns
    col1, col2, col3 = st.columns(3)
    
    # Dropdowns for selecting Kind/Cut and Cluster
    with col1:
        selected_groupby_column = st.selectbox("Select Grouping Column (Kind or Cut)", ['Kind', 'Cut'])
    
    with col2:
        selected_cluster_for_recommendation = st.selectbox("Select Cluster for Recommendation", data_cluster['Cluster'].unique())
    
    with col3:
        top_n_items = st.slider("Select Number of Items to Display", 1, 20, 5)

    # Filter data for the selected cluster
    cluster_data = data_cluster[data_cluster['Cluster'] == selected_cluster_for_recommendation]
    
    # Group data by selected column and calculate sum of 'Net Amount'
    grouped_data = cluster_data.groupby(selected_groupby_column)['Net Amount'].sum().reset_index()
    
    # Get top n items based on Net Amount sum
    top_n_items_df = grouped_data.nlargest(top_n_items, 'Net Amount')
    
    # Display the recommendation results in a DataFrame
    st.write("Top", top_n_items, f"{selected_groupby_column} Recommended for Cluster {selected_cluster_for_recommendation} based on interest:")
    st.dataframe(top_n_items_df)
    
    ## Creating the Customizable plot options:
    # Add a small title to the sidebar
    st.sidebar.markdown("**(2)Create Additional Plot:**")
    
    # Sidebar with dropdowns to select columns and cluster
    selected_x_column = st.sidebar.selectbox("Select X-axis Column", data_cluster.columns)
    selected_y_column = st.sidebar.selectbox("Select Y-axis Column", data_cluster.columns)
    selected_cluster = st.sidebar.selectbox("Select Cluster", data_cluster['Cluster'].unique())
    selected_plot_type = st.sidebar.selectbox("Select Plot Type", ["Bar", "Line", "Scatter"])

    # Display a header above the DataFrame with smaller font size
    st.markdown("<h3 style='font-size: 18px;'>Custom Visual:</h3>", unsafe_allow_html=True)
     
    # Customized Plot based on selected plot type
    # Button to show pair plot
    if st.button("Show Additional Created Plot"):
        with st.container():
            plt.figure(figsize=(8, 6))
            scatter_data = data_cluster[data_cluster['Cluster'] == selected_cluster]
        
            if selected_plot_type == "Bar":
                if scatter_data[selected_y_column].dtype == 'float64' and scatter_data[selected_x_column].dtype == 'O':
                    y_values = scatter_data.groupby(selected_x_column)[selected_y_column].sum()  # Calculate sum instead of mean
                    x_values = y_values.index
                    plt.bar(x_values, y_values, color=cluster_colors[selected_cluster])
                    plt.xticks(rotation=90)
                else:
                    plt.bar(scatter_data[selected_x_column], scatter_data[selected_y_column], color=cluster_colors[selected_cluster])
            elif selected_plot_type == "Line":
                if scatter_data[selected_y_column].dtype == 'float64' and scatter_data[selected_x_column].dtype == 'O':
                    y_values = scatter_data.groupby(selected_x_column)[selected_y_column].sum()  # Calculate sum instead of mean
                    x_values = y_values.index
                    plt.plot(x_values, y_values, marker='o', color=cluster_colors[selected_cluster])
                    plt.xticks(rotation=90)
                else:
                    plt.plot(scatter_data[selected_x_column], scatter_data[selected_y_column], marker='o', color=cluster_colors[selected_cluster])
            elif selected_plot_type == "Scatter":
                plt.scatter(scatter_data[selected_x_column], scatter_data[selected_y_column], color=cluster_colors[selected_cluster], label=f"Cluster {selected_cluster}")
        
            plt.xlabel(selected_x_column)
            plt.ylabel(selected_y_column)
            plt.title(f'Sum {selected_y_column} vs {selected_x_column} for Cluster {selected_cluster}')
            plt.legend()
            st.pyplot(plt)
    
    
    
    ## Market Basket Engine
    st.markdown("## (3) Market Basket Engine")
    # Filter sales data
    data_sales = data_cluster[data_cluster['Invoice Type'] == 'Sales']
    
    # Display a header above the DataFrame with smaller font size
    st.markdown("<h3 style='font-size: 18px;'>Select Market Basket Rules:</h3>", unsafe_allow_html=True)
    
    # Arrange the input elements next to each other using columns
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Filter by Cluster
    with col1:
        selected_cluster_market = st.selectbox("Select Cluster", data_cluster['Cluster'].unique(), key="cluster_select")
    data_selected = data_sales[data_sales['Cluster'] == selected_cluster_market]
    
    # Relevant product columns
    with col2:
        product_columns = ['Kind', 'Cut', 'Product Code', 'Product Name', 'Brand', 'Color', 'Size']
        selected_product_columns = st.selectbox('Select Product Feature', product_columns, key="product_feature_select")
    
    # Input thresholds
    with col3:
        min_support_text = st.text_input('Minimum Support', '0.010', key="min_support_input")
        min_support = float(min_support_text)
    
    with col4:
        min_confidence = st.slider('Minimum Confidence', 0.0, 1.0, 0.05, key="min_confidence_slider")
    with col5:
        min_lift = st.slider('Minimum Lift', 1.0, 10.0, 2.0, key="min_lift_slider")
        min_length = 2  # Always fixed to 2 in your case
    
    # Group transactions based on selected product columns
    transaction = data_selected.groupby(['Client', 'Date'])[selected_product_columns].apply(lambda x: frozenset(x.dropna().values.flatten()))
    
    # Apply Apriori algorithm
    rules = apriori(transaction.tolist(), min_support=min_support, min_confidence=min_confidence, min_lift=min_lift, min_length=min_length)
    results = list(rules)
    
    # Create DataFrame from inspection
    def inspect(results):
        antecedent = [tuple(result[2][0][0])[0] for result in results]
        consequent = [tuple(result[2][0][1])[0] for result in results]
        supports = [result[1] for result in results]
        confidences = [result[2][0][2] for result in results]
        lifts = [result[2][0][3] for result in results]
        return list(zip(antecedent, consequent, supports, confidences, lifts))
    
    basket_df = pd.DataFrame(inspect(results), columns=['Antecedent', 'Consequent', 'Support', 'Confidence', 'Lift'])
    
    # Display a header above the DataFrame with smaller font size
    st.write("Basket Items based on selected rules:</h3>", unsafe_allow_html=True)
    
    # Display the resulting DataFrame
    st.write(basket_df)








    
    # Display a header above the DataFrame with smaller font size
#     st.markdown("<h3 style='font-size: 18px;'>Cluster Details:</h3>", unsafe_allow_html=True)
#     ##Take the cluster statistics dictionary and denerate text description through openai api
#     # Calculate cluster statistics
#     cluster_stats = calculate_cluster_statistics(data_cluster, n_clusters)
    
#     # Generate cluster descriptions using prompt engineering
#     descriptions = generate_cluster_descriptions(cluster_stats)
    
#     # Display descriptions
#     for cluster, description in descriptions.items():
#         print(f"{cluster}: \"{description}\"")






# def calculate_cluster_statistics(cluster_data, n_clusters):
#     statistics_dict = {}

#     numeric_columns = ['Net Amount', 'Qty', 'Product_Price', 'Discount_percent']
    
#     # Look at the distribution of the features within each cluster
#     for i in range(n_clusters):
#         cluster_stats = {}
#         for column in numeric_columns:
#             cluster_stats[column] = cluster_data[cluster_data['Cluster'] == i][column].describe()
#         statistics_dict[f'Cluster {i}'] = cluster_stats

#     # Look at the correlations between the numeric features within each cluster
#     for i in range(n_clusters):
#         numeric_cluster_data = cluster_data[cluster_data['Cluster'] == i][numeric_columns]
#         cluster_corr = numeric_cluster_data.corr()
#         statistics_dict[f'Cluster {i} Correlations'] = cluster_corr

#     # Compare the customer segments to each other
#     for i in range(n_clusters):
#         for j in range(n_clusters):
#             if i != j:
#                 comparison_stats = {}
#                 comparison_stats['Net Amount'] = (
#                     cluster_data[cluster_data['Cluster'] == i]['Net Amount'].mean(),
#                     cluster_data[cluster_data['Cluster'] == j]['Net Amount'].mean()
#                 )
#                 # ... (Rest of the comparisons)
#                 statistics_dict[f'Cluster {i} vs. Cluster {j} Comparisons'] = comparison_stats

#     return statistics_dict



# # Example prompt for describing statistics
# def generate_description(cluster_name, stats):
#     prompt = f"Describe {cluster_name} based on the following statistics:\n\n"
#     for feature, feature_stats in stats.items():
#         if isinstance(feature_stats, tuple):
#             prompt += f"- {feature}: Mean {feature_stats[0]:.2f} vs. {feature_stats[1]:.2f}, "
#         elif isinstance(feature_stats, pd.Series):
#             prompt += f"- {feature}: Mean {feature_stats['mean']:.2f}, "
#         else:
#             prompt += f"- {feature}, "
#     return prompt

# # Generate descriptions using prompt engineering and GPT-3
# def generate_cluster_descriptions(cluster_stats):
#     descriptions = {}
#     for cluster, stats in cluster_stats.items():
#         prompt = generate_description(cluster, stats)
#         response = openai.Completion.create(
#             engine="davinci",
#             prompt=prompt,
#             max_tokens=150,
#         )
#         description = response.choices[0].text.strip()
#         descriptions[cluster] = description
#     return descriptions






# Add logo image to the sidebar
logo_image = "data:image/jpeg;base64,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"  # Replace with the actual path to your logo image
st.sidebar.image(logo_image, width=75)



# Tab layout
menu_data = [
    {'label': "Demographics Segmentation", 'icon': '🗺️'},
    {'label': 'Purchasing Behavior Segmentation', 'icon': '📊'},
]

dark_blue_color = 'rgb(0, 0, 128)'  # Change this to your desired dark blue color

# Add custom CSS to adjust the menu bar and page width
st.markdown(
    f"""
    <style>
    .nav-bar {{
        background-color: {dark_blue_color} !important;
        width: 100%;
    }}
    .nav-bar li {{
        width: 50%;
    }}
    .block-container {{
        max-width: 1600px;  /* Adjust the maximum width as needed */
        padding: 2rem;
        margin: 0 auto;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Apply the custom CSS to adjust the page layout
st.markdown('<div class="block-container">', unsafe_allow_html=True)

selected_menu = hc.nav_bar(
    menu_definition=menu_data,
    override_theme={
        'txc_inactive': 'white',
        'option_active': 'white'
    },
    hide_streamlit_markers=True,
    sticky_nav=True,
    sticky_mode='sticky'
)

# Display the selected section/page based on the selected menu
if selected_menu == "Demographics Segmentation":
    Demographics_Segmentation()
elif selected_menu == "Purchasing Behavior Segmentation":
    Purchasing_Behavior()

# Close the custom block-container
st.markdown('</div>', unsafe_allow_html=True)
# Footer
st.write("© Marino Lucci")


