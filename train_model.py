# train_model.py
# Description: Generates a physically-realistic dataset for transmission lines,
#              trains a high-accuracy XGBoost model, and saves it to a file.

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
import joblib
import argparse
from tqdm import tqdm

def calculate_transmission_line_params(R, L, G, C, f):
    """Calculates secondary transmission line parameters from primary ones."""
    omega = 2 * np.pi * f
    
    # Propagation Constant (gamma = alpha + j*beta)
    gamma = np.sqrt((R + 1j * omega * L) * (G + 1j * omega * C))
    alpha = np.real(gamma)  # Attenuation constant (Np/m)
    beta = np.imag(gamma)   # Phase constant (rad/m)
    
    # Characteristic Impedance (Z0)
    Z0 = np.sqrt((R + 1j * omega * L) / (G + 1j * omega * C))
    Z0_real = np.real(Z0)
    Z0_imag = np.imag(Z0)
    
    # Phase Velocity (vp)
    # Avoid division by zero if beta is zero (DC case)
    vp = omega / beta if beta != 0 else np.inf
    
    # Wavelength (lambda)
    wavelength = 2 * np.pi / beta if beta != 0 else np.inf
    
    return alpha, beta, Z0_real, Z0_imag, vp, wavelength

def generate_dataset(num_samples):
    """Generates a diverse, physically-realistic dataset."""
    print(f"Generating {num_samples} data points...")
    data = []
    
    # Define wide logarithmic ranges for realistic component values
    # Resistors (Ohms/m), Inductors (Henry/m), Conductors (Siemens/m), Capacitors (Farad/m)
    R_range = 10**np.random.uniform(-3, 2, num_samples)  # 0.001 to 100 Ohm/m
    L_range = 10**np.random.uniform(-7, -5, num_samples)  # 100 nH/m to 10 uH/m
    G_range = 10**np.random.uniform(-9, -3, num_samples)  # 1 nS/m to 1 uS/m
    C_range = 10**np.random.uniform(-12, -10, num_samples) # 1 pF/m to 100 pF/m
    f_range = 10**np.random.uniform(3, 9, num_samples)    # 1 kHz to 1 GHz

    for i in tqdm(range(num_samples), desc="Creating Dataset"):
        R, L, G, C, f = R_range[i], L_range[i], G_range[i], C_range[i], f_range[i]
        
        # Calculate outputs
        outputs = calculate_transmission_line_params(R, L, G, C, f)
        
        # Store inputs and outputs
        row = {'R': R, 'L': L, 'G': G, 'C': C, 'f': f,
               'alpha': outputs[0], 'beta': outputs[1], 
               'Z0_real': outputs[2], 'Z0_imag': outputs[3],
               'vp': outputs[4], 'lambda': outputs[5]}
        data.append(row)
        
    df = pd.DataFrame(data)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

def main(args):
    """Main function to run the data generation, training, and saving process."""
    # 1. Generate Data
    dataset = generate_dataset(args.samples)
    
    # 2. Define Features (X) and Targets (y)
    features = ['R', 'L', 'G', 'C', 'f']
    targets = ['alpha', 'beta', 'Z0_real', 'Z0_imag', 'vp', 'lambda']
    X = dataset[features]
    y = dataset[targets]
    
    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Create and Train the Model
    print("\nTraining the XGBoost model...")
    # Use MultiOutputRegressor to wrap XGBoost for predicting multiple targets
    # Using tuned parameters for high accuracy
    xgb_reg = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1 # Use all available CPU cores
    )
    
    model = MultiOutputRegressor(estimator=xgb_reg)
    model.fit(X_train, y_train)
    
    # 5. Evaluate the Model
    print("Evaluating model performance...")
    predictions = model.predict(X_test)
    score = r2_score(y_test, predictions)
    print(f"\nModel R^2 Score: {score:.4f}")
    if score > 0.97:
        print("✅ Success! Model accuracy is above 97%.")
    else:
        print("⚠️ Model accuracy is below 97%. Consider increasing dataset size or tuning parameters.")

    # 6. Save the trained model
    print(f"Saving model to '{args.output}'...")
    joblib.dump(model, args.output)
    print("Model training complete and saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a transmission line analysis model.")
    parser.add_argument("--samples", type=int, default=50000, help="Number of data samples to generate.")
    parser.add_argument("--output", type=str, default="transmission_line_model.joblib", help="Filename for the saved model.")
    args = parser.parse_args()
    main(args)