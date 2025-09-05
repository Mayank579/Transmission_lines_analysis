# analysis_gui.py
# Description: A graphical user interface to analyze transmission lines.
#              It loads a pre-trained model to provide predictions alongside
#              exact physics-based calculations and wave visualizations.

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- Physics Calculation Core (Copied from training script for consistency) ---
def calculate_transmission_line_params(R, L, G, C, f):
    """Calculates secondary transmission line parameters from primary ones."""
    omega = 2 * np.pi * f
    gamma = np.sqrt((R + 1j * omega * L) * (G + 1j * omega * C))
    alpha = np.real(gamma)
    beta = np.imag(gamma)
    Z0 = np.sqrt((R + 1j * omega * L) / (G + 1j * omega * C))
    Z0_real = np.real(Z0)
    Z0_imag = np.imag(Z0)
    vp = omega / beta if beta != 0 else 0
    wavelength = 2 * np.pi / beta if beta != 0 else 0
    return alpha, beta, Z0, vp, wavelength

# --- Main Application Class ---
class TlAnalyzerApp(tk.Tk):
    def __init__(self, model_path="transmission_line_model.joblib"):
        super().__init__()
        self.title("Transmission Line Analysis & Prediction Suite")
        self.geometry("1000x700")

        # Load the model
        try:
            self.model = joblib.load(model_path)
        except FileNotFoundError:
            messagebox.showerror("Error", f"Model file not found at '{model_path}'.\nPlease run train_model.py first.")
            self.destroy()
            return

        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Input Frame ---
        input_frame = ttk.LabelFrame(main_frame, text="Primary Inputs", padding="10")
        input_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.inputs = {}
        input_params = {
            "R (Ω/m)": "1e-2", "L (H/m)": "2.5e-7", "G (S/m)": "1e-8", "C (F/m)": "1e-10",
            "Frequency (Hz)": "1e8", "Line Length (m)": "2.0", "Load ZL (Ω)": "100+0j"
        }
        for i, (text, val) in enumerate(input_params.items()):
            label = ttk.Label(input_frame, text=text)
            label.grid(row=i, column=0, sticky="w", padx=5, pady=2)
            entry = ttk.Entry(input_frame, width=15)
            entry.grid(row=i, column=1, sticky="ew", padx=5, pady=2)
            entry.insert(0, val)
            self.inputs[text.split(" ")[0]] = entry

        calc_button = ttk.Button(input_frame, text="Calculate & Predict", command=self.on_calculate)
        calc_button.grid(row=len(input_params), columnspan=2, pady=10)

        # --- Output Frame ---
        output_frame = ttk.LabelFrame(main_frame, text="Derived Outputs", padding="10")
        output_frame.grid(row=0, column=1, padx=5, pady=5, sticky="ns")
        
        ttk.Label(output_frame, text="Parameter", font=('Helvetica', 10, 'bold')).grid(row=0, column=0, sticky='w')
        ttk.Label(output_frame, text="Exact Calculation", font=('Helvetica', 10, 'bold')).grid(row=0, column=1, sticky='w', padx=10)
        ttk.Label(output_frame, text="AI Prediction", font=('Helvetica', 10, 'bold')).grid(row=0, column=2, sticky='w', padx=10)
        
        self.outputs = {}
        output_labels = ["α (Np/m)", "β (rad/m)", "Z₀ (Ω)", "vₚ (m/s)", "λ (m)", "Γ (Load)"]
        for i, label_text in enumerate(output_labels):
            ttk.Label(output_frame, text=label_text).grid(row=i+1, column=0, sticky="w", pady=3)
            exact_val = ttk.Label(output_frame, text="-", width=25, anchor="w")
            exact_val.grid(row=i+1, column=1, sticky="w", padx=10)
            pred_val = ttk.Label(output_frame, text="-", width=25, anchor="w")
            pred_val.grid(row=i+1, column=2, sticky="w", padx=10)
            self.outputs[label_text.split(" ")[0]] = (exact_val, pred_val)

        # --- Plot Frame ---
        plot_frame = ttk.LabelFrame(main_frame, text="Voltage Wave Visualization", padding="10")
        plot_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)

        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.ax.set_title("Voltage Magnitude Along Line")
        self.ax.set_xlabel("Distance from Load, z' (m)")
        self.ax.set_ylabel("|V(z')| (Volts)")
        self.ax.grid(True)
        self.fig.tight_layout()

    def on_calculate(self):
        try:
            # 1. Get inputs from GUI
            values = {key: float(eval(entry.get())) for key, entry in self.inputs.items()}
            R, L, G, C, f = values['R'], values['L'], values['G'], values['C'], values['Frequency']
            line_len = values['Line']
            ZL = complex(self.inputs['Load'].get())

            # 2. Perform Exact Calculation
            alpha, beta, Z0, vp, wavelength = calculate_transmission_line_params(R, L, G, C, f)
            gamma = alpha + 1j * beta
            Gamma_L = (ZL - Z0) / (ZL + Z0)

            # 3. Perform AI Prediction
            input_df = pd.DataFrame([[R, L, G, C, f]], columns=['R', 'L', 'G', 'C', 'f'])
            pred = self.model.predict(input_df)[0]
            p_alpha, p_beta, p_Z0_real, p_Z0_imag, p_vp, p_lambda = pred
            p_Z0 = p_Z0_real + 1j * p_Z0_imag
            
            # 4. Update Output Labels
            self.outputs['α'][0].config(text=f"{alpha:.4e}")
            self.outputs['β'][0].config(text=f"{beta:.4f}")
            self.outputs['Z₀'][0].config(text=f"{Z0.real:.2f} + {Z0.imag:.2f}j")
            self.outputs['vₚ'][0].config(text=f"{vp:.4e}")
            self.outputs['λ'][0].config(text=f"{wavelength:.4f}")
            self.outputs['Γ'][0].config(text=f"{Gamma_L.real:.3f} + {Gamma_L.imag:.3f}j")

            self.outputs['α'][1].config(text=f"{p_alpha:.4e}")
            self.outputs['β'][1].config(text=f"{p_beta:.4f}")
            self.outputs['Z₀'][1].config(text=f"{p_Z0.real:.2f} + {p_Z0.imag:.2f}j")
            self.outputs['vₚ'][1].config(text=f"{p_vp:.4e}")
            self.outputs['λ'][1].config(text=f"{p_lambda:.4f}")
            self.outputs['Γ'][1].config(text="N/A (Not Predicted)")

            # 5. Update Plot
            self.update_plot(gamma, Gamma_L, line_len)

        except Exception as e:
            messagebox.showerror("Input Error", f"Could not perform calculation.\nPlease check your inputs.\nError: {e}")

    def update_plot(self, gamma, Gamma_L, length, V_plus=1.0):
        self.ax.clear()
        z_prime = np.linspace(0, length, 400) # z' = l - z (distance from load)
        
        # V(z') = V+ * e^(gamma*z') * (1 + Gamma_L * e^(-2*gamma*z'))
        voltage_wave = V_plus * np.exp(gamma * z_prime) * (1 + Gamma_L * np.exp(-2 * gamma * z_prime))
        
        self.ax.plot(z_prime, np.abs(voltage_wave), color='b')
        self.ax.set_title("Voltage Magnitude Along Line")
        self.ax.set_xlabel("Distance from Load, z' (m)")
        self.ax.set_ylabel("|V(z')| (Volts)")
        self.ax.grid(True)
        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    app = TlAnalyzerApp()
    app.mainloop()