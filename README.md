# Changepoint-Detection-in-the-Presence-of-Outliers

This repository contains a Machine Learning for Time Series project as part of the MVA Master. The project focuses on the R-FPOP algorithm, demonstrating how bounded loss functions can accurately detect structural changes in time series data while remaining robust to extreme outliers.

## For Developers

## Dependencies

To isolate the project dependencies, it is recommended to use a virtual environment.
- To create the virtual environment: `python -m venv venv`
- To activate it: `source venv/bin/activate`
- To install the required dependencies: `pip install -r requirements.txt`

## Run With Docker

- Build the image:
	- `docker build -t rfpop-streamlit-app .`
- Run the container:
	- `docker run --rm -p 8501:8501 rfpop-streamlit-app`
- Open the app:
	- `http://localhost:8501`

By default, container logs display `http://localhost:8501`.
If you map a different host port, open that mapped port in your browser.

### Helper Script For Correct URL Display

Use the helper script to build and run the container while automatically setting
the Streamlit displayed URL to your chosen host port.

- Make it executable once:
	- `chmod +x scripts/run_docker.sh`
- Run on default port `8501`:
	- `scripts/run_docker.sh`
- Run on a custom port (example `8504`):
	- `scripts/run_docker.sh 8504`

## Toy Data From SSPCloud S3 (Public MinIO)

The app can load built-in toy CSV examples directly from public MinIO storage.

- Default base URL:
	- `https://minio.lab.sspcloud.fr/asicard/MPPDS - Projet`
- The app uses these toy filenames:
	- `data example 1.csv`
	- `data example 2.csv`
	- `data example 3.csv`
	- `data example 4.csv`
- If remote loading fails, the app falls back to local files in `data/` when available.

You can override the default base URL with:
- `TOY_DATA_BASE_URL`

### Pre-commit

Pre-commit automatically formats your code before each commit, ensuring that all developers follow the same formatting rules. To install it:
- Install Pre-commit: `pip install pre-commit`
- Set up Pre-commit in your project: `pre-commit install`
Once installed, Pre-commit will automatically run the defined checks and formatting before each commit.

### Reference Paper
Fearnhead, P., & Rigaill, G. (2019). Changepoint Detection in the Presence of Outliers. Journal of the American Statistical Association, 114(525), 169–183.

### Implemented Methods and experiments
The repository implements the dynamic programming algorithm described in the paper with three distinct cost functions (L2 loss, Huber loss, Biweight loss) to compare sensitivity and robustness.

The analysis is performed on two types of datasets:
- Simulated Scenarios: Reproduction of the six benchmark scenarios described in the article (varying noise levels, Student-t noise, short segments) to validate the theoretical properties of the Biweight loss.
- Real-world Economic Indicators: Application of the algorithms to financial time series from the FRED database, including: inflation expectations, GDP growth rates (Japan, UK, Germany), market volatility and credit spreads.

### Results
The results of our experiments are available in the "final_notebook.ipynb" notebook. A detailed analysis is also available in the "Report of the project.pdf" file.

### Requirements
Python 3.x

numpy

pandas

matplotlib

statsmodels
