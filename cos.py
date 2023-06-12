import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import plotly.express as px
from sklearn.metrics import classification_report

# Load the cleaned dataset
df = pd.read_csv('Telco-Customer-Churn-dataset-cleaned.csv')

# Encode categorical attributes to integers
encoder = LabelEncoder()
categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
                       'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                       'TechSupport', 'StreamingTV', 'StreamingMovies', 'InternetService',
                       'Contract', 'PaymentMethod', 'tenure-binned', 'MonthlyCharges-binned',
                       'TotalCharges-binned']

for column in categorical_columns:
    df[column] = encoder.fit_transform(df[column])

# Sidebar for clustering options
st.sidebar.title("Clustering Options")
clustering_algorithm = st.sidebar.selectbox("Clustering Algorithm", ["K-means", "Agglomerative", "DBSCAN"])

if clustering_algorithm == "K-means":
    num_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=3)
    algorithm = KMeans(n_clusters=num_clusters, random_state=42)
elif clustering_algorithm == "Agglomerative":
    num_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=3)
    algorithm = AgglomerativeClustering(n_clusters=num_clusters)
elif clustering_algorithm == "DBSCAN":
    eps = st.sidebar.slider("EPS", min_value=0.1, max_value=2.0, value=0.5)
    algorithm = DBSCAN(eps=eps)

# Perform clustering
X_clustering = df[['MonthlyCharges', 'TotalCharges']]
algorithm.fit(X_clustering)
df['Cluster'] = algorithm.labels_

# Sidebar for prediction options
st.sidebar.title("Prediction Options")
prediction_algorithm = st.sidebar.selectbox("Prediction Algorithm", ["Random Forest", "K-Nearest Neighbors", "Naive Bayes", "Decision Tree"])

# Preprocessing for prediction
X_prediction = df.drop('Churn', axis=1)
y_prediction = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X_prediction, y_prediction, test_size=0.2, random_state=42)

# Model selection for prediction
if prediction_algorithm == "Random Forest":
    model = RandomForestClassifier()
elif prediction_algorithm == "K-Nearest Neighbors":
    model = KNeighborsClassifier()
elif prediction_algorithm == "Naive Bayes":
    model = GaussianNB()
elif prediction_algorithm == "Decision Tree":
    model = DecisionTreeClassifier()

# Main app
st.title("Customer Churn App")

# Button to show/hide data and description
df2 = pd.read_csv('Telco-Customer-Churn.csv')
show_datadf2 = st.button("Show Raw Data")
if show_datadf2:
    st.write("Dataset before cleaning:")
    st.dataframe(df2)
    if st.button("Close"):
        show_datadf2 = False

# Button to show/hide data and description
df3 = pd.read_csv('Telco-Customer-Churn-dataset-cleaned.csv')
show_datadf3 = st.button("Show Cleaned Data")
if show_datadf3:
    st.write("Dataset after cleaning:")
    st.dataframe(df3)
    if st.button("Close"):
        show_datadf3 = False

# Button to generate data
generate_button = st.sidebar.button("Generate Data")
if generate_button:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Clustering results
    st.write("Clustering Results:")
    st.write("Cluster Sizes:")
    st.dataframe(df['Cluster'].value_counts())

    # Plot the clusters using Plotly Express
    fig = px.scatter(df, x='MonthlyCharges', y='TotalCharges', color='Cluster', labels={'MonthlyCharges': 'Monthly Charges', 'TotalCharges': 'Total Charges'}, title='Customer Clusters')
    st.plotly_chart(fig)

    # Prediction results
    st.write("Prediction Results:")
    st.write("Accuracy: ", model.score(X_test, y_test))
    st.write("Predicted Churn: ", y_pred[:10])
    st.write("Classification Report:")
    st.write(classification_report(y_test, y_pred))


# Documentation
st.title("Documentation")
st.write("This application performs customer churn analysis using clustering and prediction algorithms.")
st.write("The cleaned dataset is encoded to convert categorical attributes into integers before clustering and prediction.")
st.write("The user can select the clustering algorithm and adjust the algorithm-specific parameters using the sidebar.")
st.write("The selected clustering algorithm is applied to identify clusters of customers based on their monthly charges and total charges.")
st.write("The clusters are displayed in a table and visualized in an interactive scatter plot.")
st.write("The user can also select the prediction algorithm for churn prediction.")
st.write("The selected algorithm is trained on the training data, and predictions are made on the testing data.")
st.write("The accuracy of the model and the predicted churn values are displayed.")
st.write("The documentation provides an overview of the application and explains the data attributes, preprocessing, clustering process, model selection, and prediction process.")
