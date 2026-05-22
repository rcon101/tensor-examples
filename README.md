# TensorFlow Character Recognition App

This project trains a TensorFlow model to recognize single A-Z characters and serves it through a small Flask web app. Upload a JPG or PNG containing one character, and the app returns the most likely letter.

## Setup

```bash
sudo apt install python3
python3 --version
sudo apt install python3.<your_python_version>-venv

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train the model

```bash
python train.py --epochs 8
```

The trained model is saved to `models/character_model.keras`. Training uses EMNIST handwritten letters plus locally rendered font samples, so it handles both handwriting-style characters and simple typed characters better than EMNIST alone.

## Run the app

```bash
python app.py
```

Open `http://127.0.0.1:5000` and upload a JPG or PNG with a single handwritten letter.

## Notes

- The model works best on centered, high-contrast single letters.
- The app preprocesses images by converting them to grayscale, detecting foreground/background polarity, cropping the visible character, centering it, resizing to 28x28, and normalizing pixel values.
- For printed text or full-word OCR, use an OCR engine such as Tesseract or a sequence model instead of this single-character classifier.
