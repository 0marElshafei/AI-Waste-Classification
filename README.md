# AI Waste Classification

A real-time computer vision system that classifies waste items from a live webcam
feed and tells the user which bin to use: Landfill, Organic, Paper Recycling, or
Container Recycling.

Built during the University of Toronto CREATE "AI: Sorting it Out!" summer design
program.

## How it works

- A model was trained on a [Teachable Machine](https://teachablemachine.withgoogle.com/)
  image project using labeled photos of 12 common waste items (egg cartons, cans,
  plastic bottles, cardboard, masks, etc.).
- The trained Keras model (`keras_model.h5`) and its labels (`labels.txt`) are loaded
  at runtime.
- Each webcam frame is resized and normalized to match the model's expected input,
  then classified.
- A prediction is only trusted, and mapped to a disposal bin, above a 67% confidence
  threshold, to avoid flagging a bin from a low-confidence guess.
- The predicted item, confidence score, and correct bin are overlaid live on the
  video feed.

## Run it

```bash
pip install -r requirements.txt
python waste_classifier.py
```

Requires a webcam. Press `Esc` to quit.

## Files

- `waste_classifier.py` — main script: loads the model, runs live classification, maps predictions to disposal bins
- `keras_model.h5` — trained Teachable Machine model
- `labels.txt` — class labels used by the model
- `requirements.txt` — Python dependencies

## Notes

The model-loading step includes a small compatibility fix for a Keras/TensorFlow
versioning quirk in older Teachable Machine exports (stripping a deprecated
`"groups": 1,` key from the saved model config).
