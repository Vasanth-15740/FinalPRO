import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import streamlit as st
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import joblib
import os

# Load encoders and vectorizer
try:
    le_user = joblib.load('../rumor_pheme/le_user.pkl')
    le_topic = joblib.load('../rumor_pheme/le_topic.pkl')
    le_is_rumor = joblib.load('../rumor_pheme/le_is_rumor.pkl')
    vectorizer = joblib.load('../rumor_pheme/tfidf_vectorizer.pkl')

    # Ensure "unknown" is part of the label encoders
    for le in [le_user, le_topic, le_is_rumor]:
        if "unknown" not in le.classes_:
            le.classes_ = np.append(le.classes_, "unknown")

except FileNotFoundError as e:
    st.error(f"A required file (encoder or vectorizer) was not found: {e}")
    st.stop()

# Define the GCN model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Load model function with dynamic adaptation
def load_model(model, model_path):
    checkpoint = torch.load(model_path)
    model_dict = model.state_dict()
    
    # Check if output layer size matches
    if checkpoint['conv2.lin.weight'].shape[0] != model_dict['conv2.lin.weight'].shape[0]:
        st.warning("⚠️ Output size mismatch! Adjusting the model to match saved checkpoint.")
        
        # Adjust out_channels dynamically
        out_channels = checkpoint['conv2.lin.weight'].shape[0]
        model = GCN(model_dict['conv1.lin.weight'].shape[1], 64, out_channels)
    
    model.load_state_dict(checkpoint)
    return model

# Preprocessing function
def preprocess_data_for_prediction(df):
    df['user.handle'] = df['user.handle'].apply(lambda x: x if x in le_user.classes_ else "unknown")
    df['user.handle'] = le_user.transform(df['user.handle'])

    df['topic'] = df['topic'].apply(lambda x: x if x in le_topic.classes_ else "unknown")
    df['topic'] = le_topic.transform(df['topic'])

    # Convert text using TF-IDF vectorizer
    X_text = vectorizer.transform(df['text']).toarray()

    # Combine categorical and text features
    categorical_features = df[['user.handle', 'topic']].values
    x = np.hstack([categorical_features, X_text])
    return torch.tensor(x, dtype=torch.float)

# Prediction function
def predict(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data)
        _, predicted = torch.max(out, dim=1)
    return predicted

# Edge index creation function
def create_edge_index(df):
    edges = []
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if df['topic'].iloc[i] == df['topic'].iloc[j]:
                edges.append([i, j])
                edges.append([j, i])  # For undirected graph
    return torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.tensor([[],[]], dtype=torch.long)

# Streamlit app
def main():
    st.title("Rumor Detection using GCN")

    uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            # Ensure the necessary columns exist
            required_columns = {'user.handle', 'topic', 'text', 'is_rumor'}
            if not required_columns.issubset(df.columns):
                st.error(f"Uploaded CSV is missing required columns: {required_columns - set(df.columns)}")
                return

            # Drop rows with missing 'is_rumor' values
            df = df.dropna(subset=['is_rumor'])
            df['is_rumor'] = df['is_rumor'].astype(str)  # Convert to string for encoding

            # Handle unknown labels in is_rumor safely
            df['is_rumor'] = df['is_rumor'].apply(lambda x: x if x in le_is_rumor.classes_ else "unknown")
            df['is_rumor'] = le_is_rumor.transform(df['is_rumor'])
            labels = df['is_rumor'].values

            # Preprocess input features
            features_tensor = preprocess_data_for_prediction(df)

            # Initialize GCN model dynamically
            in_channels = features_tensor.shape[1]
            hidden_channels = 64
            out_channels = len(le_is_rumor.classes_)  # Get number of classes from encoder

            model = GCN(in_channels, hidden_channels, out_channels)

            model_path = '../rumor_pheme/gnn_model.pth'
            if os.path.exists(model_path):
                model = load_model(model, model_path)
            else:
                st.error(f"Model file not found at: {model_path}")
                return

            # Create graph structure
            edge_index = create_edge_index(df)  
            data = Data(x=features_tensor, edge_index=edge_index, y=torch.tensor(labels, dtype=torch.long))

            # Get predictions
            predictions = predict(model, data)

            # Compute accuracy
            true_labels = df['is_rumor'].values
            accuracy = accuracy_score(true_labels, predictions.numpy())
            st.write(f"Model Accuracy: {accuracy:.4f}")

            # Display predictions
            prediction_results = pd.DataFrame({
                'ID': df.index,
                'True Label': true_labels,
                'Predicted': predictions.numpy()
            })
            st.write(prediction_results)

        except pd.errors.EmptyDataError:
            st.error("Uploaded file is empty.")
        except pd.errors.ParserError:
            st.error("Error parsing the CSV file. Please check the format.")
        except FileNotFoundError as e:
            st.error(f"A required file (model, encoder, vectorizer) was not found: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

    else:
        st.write("Please upload a CSV file to begin.")

if __name__ == "__main__":
    main()
