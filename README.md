# IMDB Movie Review Sentiment Analysis (SimpleRNN)

A binary sentiment classifier that labels IMDB movie reviews as **Positive** or **Negative**, built with a SimpleRNN in TensorFlow/Keras and served through a Streamlit web app.

## Project Structure

| File | Description |
|------|-------------|
| `simplernn.ipynb` | Trains the SimpleRNN model and saves it to `simplernn_imdb.h5` |
| `prediction.ipynb` | Loads the trained model and runs predictions on example reviews |
| `main.py` | Streamlit web app for interactive sentiment prediction |
| `embedding.ipynb` | Notebook exploring word embeddings |
| `requirements.txt` | Python dependencies |

## Model

- **Embedding** layer (vocabulary size 10,000 → 128 dimensions)
- **SimpleRNN** layer (128 units, `tanh` activation)
- **Dense** output layer (1 unit, `sigmoid` activation)

Reviews are padded/truncated to a length of 500 tokens.

## Setup

```bash
# Create and activate a virtual environment (optional)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Train the model
Run all cells in `simplernn.ipynb`. This downloads the IMDB dataset, trains the model, and saves `simplernn_imdb.h5`.

### 2. Run predictions
Open `prediction.ipynb` to classify example reviews, or launch the web app:

```bash
streamlit run main.py
```

Then enter a review in the text box and click **Classify**. A prediction score above 0.5 means Positive; below means Negative.

## Tech Stack

Python · TensorFlow / Keras · Streamlit · NumPy
