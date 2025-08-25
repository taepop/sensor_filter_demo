# Sensor Data Filtering (Moving Average, Butterworth, Kalman)

This project demonstrates different filtering techniques for noisy sensor data, including:
- **Moving Average**
- **Butterworth Low-pass Filter**
- **1D Kalman Filter**
The code generates synthetic sinusoidal data with Gaussian noise, applies filters, and compares their effectiveness.

## Project Structure
sensor_filter_demo/
|-- src/main.py #main scipt with filtering methods
|-- results/ # output files (CSV, Excel, plots)
|-- requirements.txt # dependencies
|-- README.md # project description

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/taepop/sensor_filter_demo.git
   cd sensor_filter_demo
2. Create a virtual environment and install dependencies:
   python -m venv venv
   source venv/bin/activate
   venv\Scripts\activate
   pip install -r requirements.txt
3. Run the script
   python src/main.py

This will:
- Save results to results/filtered_results.csv and .xlsx
- Generate a plot in results/filter_comparison.png
