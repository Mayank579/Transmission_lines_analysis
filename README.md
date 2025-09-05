## A Decoupled Software Suite for High-Fidelity Transmission Line Analysis
This repository contains the source code for a comprehensive software suite designed for the analysis and prediction of transmission line behavior. The project's primary innovation is a decoupled architecture that separates the offline training of a machine learning model from the real-time, user-facing analysis application. This approach ensures the user tool remains fast and responsive while leveraging a powerful, accurately trained predictive model.

The suite's core is an engine built on first-principles electromagnetic theory, capable of calculating key secondary parameters (
gamma,Z_0,
alpha,
beta) from primary physical inputs (R,L,G,C,f). This is augmented by an XGBoost regression model, trained to an R² score of 0.9782, which provides instantaneous predictions for validation and rapid analysis.

Core Components
Model Training Engine (train_model.py): A command-line utility that programmatically generates a vast, physically-grounded dataset, trains the XGBoost model, and exports the final trained artifact (.joblib file).

Analysis & Prediction Tool (analysis_gui.py): A user-friendly graphical tool that loads the pre-trained model. It performs calculations, displays exact and AI-predicted results, and provides clear visualizations of the voltage wave profile.

