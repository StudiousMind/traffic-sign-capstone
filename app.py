import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import os
import torch.nn.functional as F
import google.generativeai as genai
import os
import numpy as np




# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Traffic Sign Classifier",
    page_icon="🚦",
    layout="centered"
)


# ----------------------------
# Gemini configuration (optional)
# ----------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("DEBUG: GEMINI_API_KEY present? -> True")
else:
    print("DEBUG: GEMINI_API_KEY present? -> False")

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("🚦 Traffic Sign App")
st.sidebar.markdown("""
This demo uses a deep learning model (ResNet18 fine-tuned on GTSRB)
to classify German traffic signs.

**How to use it:**
1. Upload a clear traffic sign image (preferably from the GTSRB dataset or similar).
2. Wait for the model to analyze it.
3. Review the prediction, safety note, and top-3 suggestions.

**Note:**  
The model was trained on German-style signs. Accuracy may drop on Canadian/US signs.
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**Project:** Deep Learning Capstone\n**Student:** Khaled (Kai) Balhareth\n**Year:** 2025")

# ----------------------------
# GTSRB label names (official order)
# ----------------------------
LABEL_MAP = {
    0: "Speed limit (20 km/h)",
    1: "Speed limit (30 km/h)",
    2: "Speed limit (50 km/h)",
    3: "Speed limit (60 km/h)",
    4: "Speed limit (70 km/h)",
    5: "Speed limit (80 km/h)",
    6: "End of speed limit (80 km/h)",
    7: "Speed limit (100 km/h)",
    8: "Speed limit (120 km/h)",
    9: "No passing",
    10: "No passing (trucks)",
    11: "Right of way at intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "No trucks",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Icy conditions",
    31: "Wild animals crossing",
    32: "End of all limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing (trucks)",
}

SAFETY_EXPLANATIONS = {
    "Stop": "This sign requires all vehicles to come to a complete stop to prevent collisions at intersections.",
    "Yield": "Drivers must slow down and give right-of-way to avoid obstructing traffic flow.",
    "Pedestrians": "Alerts drivers to watch for people crossing the road to prevent pedestrian accidents.",
    "No entry": "Restricts access to dangerous or restricted areas to prevent traffic flow conflicts.",
    "Speed limit (70 km/h)": "Controls vehicle speed to reduce the risk of high-impact collisions.",
    "Speed limit (30 km/h)": "Used in residential or pedestrian-heavy areas to ensure safety.",
    "No passing": "Prevents overtaking in unsafe zones like curves or hills.",
    "Turn right ahead": "Warns the driver of an upcoming turn to prepare the vehicle and avoid sudden maneuvers.",
    "Roundabout mandatory": "Informs drivers of a circular traffic pattern, reducing speed and preventing major collisions.",
    "Keep right": "Ensures correct lane usage to avoid head-on collisions.",
}

# ----------------------------
# Load model
# ----------------------------
@st.cache_resource
def load_model():
    num_classes = 43  # GTSRB classes
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)

    model.load_state_dict(torch.load("resnet18_gtsrb_state_dict.pth",
                                     map_location=torch.device("cpu")))
    model.eval()
    return model





# ----------------------------
# Image preprocessing
# ----------------------------
test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# ----------------------------
# Prediction function
# ----------------------------
def predict(image):
    img = Image.open(image).convert("RGB")
    img_tensor = test_transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)[0]  # shape [43]

    # Top-3 indices
    topk_probs, topk_indices = torch.topk(probs, k=3)

    topk_probs = topk_probs.cpu().numpy()
    topk_indices = topk_indices.cpu().numpy()

    results = []
    for idx, p in zip(topk_indices, topk_probs):
        label = LABEL_MAP.get(int(idx), f"Class {int(idx)}")
        results.append((int(idx), label, float(p)))

    return results, img


class GradCAM:
    """
    Simple Grad-CAM for ResNet18 on the last conv block (layer4[-1]).
    """

    def __init__(self, model, target_layer, transform):
        self.model = model
        self.target_layer = target_layer
        self.transform = transform
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out

        def backward_hook(module, grad_in, grad_out):
            # grad_out is a tuple; we want the gradient wrt the output
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, pil_img, class_idx=None):
        """
        pil_img: PIL.Image (original image)
        class_idx: which class index to explain (int). If None, use model's argmax.
        Returns a PIL.Image with heatmap overlay.
        """
        self.model.eval()
        self.model.zero_grad()
        self.activations = None
        self.gradients = None

        # Preprocess
        x = self.transform(pil_img).unsqueeze(0)  # [1, 3, H, W]

        # Forward with grad enabled
        x = x.requires_grad_(True)
        with torch.enable_grad():
            outputs = self.model(x)  # [1, num_classes]
            if class_idx is None:
                class_idx = outputs.argmax(dim=1).item()
            score = outputs[0, class_idx]
            score.backward()

        # Now self.activations: [1, C, h, w], self.gradients: [1, C, h, w]
        if self.activations is None or self.gradients is None:
            return pil_img  # fallback

        activations = self.activations[0]       # [C, h, w]
        gradients = self.gradients[0]           # [C, h, w]

        # Global average pooling over gradients
        weights = gradients.mean(dim=(1, 2))    # [C]

        # Weighted sum of activations
        cam = torch.zeros(activations.shape[1:], dtype=activations.dtype)
        for c, w in enumerate(weights):
            cam += w * activations[c]

        cam = torch.relu(cam)

        # Normalize between 0 and 1
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        # Upsample to image size
        cam = cam.unsqueeze(0).unsqueeze(0)  # [1,1,h,w]
        cam = torch.nn.functional.interpolate(
            cam,
            size=(pil_img.height, pil_img.width),
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().detach().cpu().numpy()  # [H, W]

        # Convert original image to numpy
        img_np = np.array(pil_img).astype("float32") / 255.0  # [H, W, 3]
        heatmap = cam[..., None]  # [H, W, 1]

        # Simple red overlay: blend red with original
        red = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # pure red
        alpha = 0.5  # heatmap strength

        overlay = img_np * (1 - alpha * heatmap) + red * (alpha * heatmap)
        overlay = np.clip(overlay, 0.0, 1.0)

        overlay_img = Image.fromarray((overlay * 255).astype("uint8"))
        return overlay_img

model = load_model()
# Grad-CAM on the last ResNet18 block
gradcam = GradCAM(model, model.layer4[-1], test_transform)



def generate_gemini_explanation(label: str):
    """
    Ask Gemini to explain this traffic sign in simple safety-focused language.
    Returns a string, or None if something goes wrong.
    """
    if not GEMINI_API_KEY:
        return None

    prompt = f"""
You are a road safety assistant. Explain the meaning and safety importance 
of this traffic sign:

Sign: "{label}"

Write 2–3 short sentences in simple English, focusing on safety for drivers and pedestrians.
"""

    try:
        # Create model object
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        # Call generate_content on the model
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print("Gemini error:", e)
        return None


# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🚦 Traffic Sign Classifier (GTSRB)")
st.markdown("""
<div style="padding:10px; border-radius:8px; background-color:#fff3cd;">
<b>⚠️ Note:</b> This model is trained on the <b>German GTSRB dataset</b>.  
Accuracy may drop when uploading traffic signs from other countries (e.g., Canada or the US) because they have different shapes, colors, and designs.
</div>
""", unsafe_allow_html=True)

with st.expander("ℹ️ About this Model"):
    st.markdown("""
    **Dataset:** German Traffic Sign Recognition Benchmark (GTSRB)  
    **Model Type:** ResNet18 fine-tuned with PyTorch  
    **Training Accuracy:** ~99%  
    **Test Accuracy:** ~97%  
    **Purpose:** Classify traffic signs for safety and automation use cases  
    **Limitation:** Performance decreases on non-German signs due to visual differences  
    """)

st.write("Upload a traffic sign image and the model will predict the class.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with st.spinner("Analyzing sign..."):
        try:
            results, img = predict(uploaded_file)
        except Exception as e:
            st.error("❌ Something went wrong while processing the image.")
            st.text(str(e))
            results, img = None, None

        if results is not None:
        # Show uploaded image
            st.image(img, caption="Uploaded Traffic Sign", width=400)

            # 🔥 Grad-CAM explanation (where the model is looking)
            try:
                cam_img = gradcam.generate(img, class_idx=results[0][0])  # best_idx
                st.image(cam_img, caption="Grad-CAM: Model Focus", width=400)
            except Exception as e:
                st.write("_Grad-CAM unavailable (error generating heatmap)._")
                st.text(str(e))

            # Best prediction
            best_idx, best_label, best_prob = results[0]
            st.markdown(
                f"### ✅ Prediction: **{best_label}**  \n"
                f"Confidence: {best_prob*100:.2f}%"
            )

        
        

                # Safety explanation
        explanation = SAFETY_EXPLANATIONS.get(
            best_label,
            "This sign is part of the GTSRB dataset and helps guide traffic flow and safety for road users."
        )
        st.markdown(f"### 🛡️ Safety Note\n{explanation}")

        # AI explanation (Gemini)
        gpt_text = generate_gemini_explanation(best_label)
        st.markdown("### 🤖 AI Explanation (Gemini)")
        if gpt_text:
            st.write(gpt_text)
        else:
            st.write("_AI explanation unavailable (no API key or an error occurred)._")

        # Top-3 predictions
        st.markdown("#### Top-3 predictions")
        for idx, label, prob in results:
            st.write(f"- {label} (Class {idx}) — **{prob*100:.2f}%**")


st.markdown("""
<hr>
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    Capstone Project • 2025<br>
    Developed by Kai Balhareth 1250916
</div>
""", unsafe_allow_html=True)
