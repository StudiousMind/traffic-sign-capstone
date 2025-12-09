# Traffic Sign Classifier (GTSRB)

This project is a capstone for my PyTorch deep learning course.  
It uses a fine-tuned ResNet18 model to classify German traffic signs (GTSRB dataset) and a Streamlit web app to demo the model.

The app also integrates Google Gemini to generate a simple safety explanation for the predicted sign and includes a Grad-CAM visualization to show where the model is "looking".

---

## Features

- ✅ ResNet18 fine-tuned on the **GTSRB** traffic sign dataset  
- ✅ Streamlit web app with image upload  
- ✅ Top-1 and Top-3 predictions with confidence scores  
- ✅ Safety note for each predicted class  
- ✅ **Gemini** integration for natural-language explanation of the sign  
- ✅ **Grad-CAM** heatmap overlay to visualize model focus  

---

## Tech Stack

- Python
- PyTorch / Torchvision
- Streamlit
- Google Generative AI (Gemini)
- NumPy, Pillow, scikit-learn

---

## Project Structure

```text
traffic_sign_capstone/
├── app.py                # Streamlit app (inference + UI)
├── best_resnet18_gtsrb.pth  # Trained PyTorch model weights
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── .gitignore
