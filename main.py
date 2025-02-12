import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import streamlit as st
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, CarliniL2Method, DeepFool
from art.estimators.classification import PyTorchClassifier
import matplotlib.pyplot as plt

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def adversarial_training(model, x_train, y_train, security_level):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(1, 28, 28),
        nb_classes=10
    )

    attack_map = {
        "Fast": FastGradientMethod(estimator=classifier, eps=0.1),
        "Low": FastGradientMethod(estimator=classifier, eps=0.3),
        "Medium": ProjectedGradientDescent(estimator=classifier, eps=0.3, max_iter=10),
        "High": ProjectedGradientDescent(estimator=classifier, eps=0.4, max_iter=20),
    }
    attack = attack_map[security_level]
    x_train_adv = attack.generate(x=x_train)
    
    x_combined = np.concatenate((x_train, x_train_adv))
    y_combined = np.concatenate((y_train, y_train))

    classifier.fit(x_combined, y_combined, batch_size=64, nb_epochs=10)
    return model

def load_custom_dataset(file_path):
    if not file_path.endswith(".csv"):
        raise ValueError("Unsupported file type. Please upload a .csv file.")

    data = pd.read_csv(file_path)

    if "label" not in data.columns:
        raise ValueError("Dataset must contain a 'label' column.")

    X = data.iloc[:, :-1].values.reshape(-1, 1, 28, 28).astype("float32") / 255.0
    y = data["label"].dropna().values
    y = torch.tensor(y, dtype=torch.long)

    return X, y

st.title("Adversarial Training System")

if not os.path.exists("uploads"):
    os.makedirs("uploads")

uploaded_model = st.file_uploader("Upload your PyTorch model (.pth or .pt)", type=["pth", "pt"])
uploaded_data = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

security_level = st.selectbox("Select Security Level", options=["Fast", "Low", "Medium", "High"])

if st.button("Start Training"):
    if uploaded_model and uploaded_data:
        try:
            model = SimpleCNN()
            model.load_state_dict(torch.load(uploaded_model, weights_only=True))
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()

        try:
            x_train, y_train = load_custom_dataset(uploaded_data)
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            st.stop()

        with st.spinner("Training in progress..."):
            robust_model = adversarial_training(model, x_train, y_train, security_level)

        torch.save(robust_model.state_dict(), "secure_model.pth")
        st.success("Training completed!")
        st.download_button("Download Secure Model", data=open("secure_model.pth", "rb"), file_name="secure_model.pth")

    else:
        st.error("Please upload both the model and dataset!")
