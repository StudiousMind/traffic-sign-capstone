import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
from openai import OpenAI
import os

#  Gemini API setup (optional)
import google.generativeai as genai
import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("No Gemini API key found.")

def generate_gemini_explanation(label: str):
    if not GEMINI_API_KEY:
        return None
    
    model = genai.GenerativeModel("gemini-pro")
    prompt = f"Explain the meaning and safety importance of this traffic sign: {label}"
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print("Gemini error:", e)
        return None

# ----------------------------




st.set_page_config(
    page_title="Traffic Sign Classifier",
    page_icon="🚦",
    layout="centered"
)
# ----------------------------

# GPT client (optional, only works if API key is set)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
gpt_client = None
if OPENAI_API_KEY:
    gpt_client = OpenAI(api_key=OPENAI_API_KEY)
# ----------------------------

# Sidebar
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
st.sidebar.markdown("**Project:** Deep Learning Capstone\n**Student:** Khaled Balhareth\n**Year:** 2025")


# ----------------------------
# GTSRB label names (official order)
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

    model.load_state_dict(torch.load("resnet18_gtsrb_state_dict.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

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
import torch.nn.functional as F

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

# GPT Step 3 — Helper function to generate explanation
def generate_gpt_explanation(label: str):
    """
    Ask GPT to explain this traffic sign in simple, safety-focused language.
    Returns a string, or None if GPT is not configured.
    """
    if gpt_client is None:
        return None  # GPT not available

    prompt = f"""
You are a road safety assistant. Explain the meaning and safety importance of this traffic sign:

Sign: "{label}"

Write 2–3 short sentences in simple English, focusing on physical safety and risk reduction for drivers and pedestrians.
"""

    try:
        response = gpt_client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
        )
        text = response.output[0].content[0].text
        return text
    except Exception as e:
        # If anything fails, just don't break the app
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
    **Model Type:** Custom CNN built with PyTorch  
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

        # Best prediction
        best_idx, best_label, best_prob = results[0]
        st.markdown(
            f"### ✅ Prediction: **{best_label}**  \n"
            f"Confidence: {best_prob*100:.2f}%"
        )

        # Safety explanation (your existing dict)
        explanation = SAFETY_EXPLANATIONS.get(
            best_label,
            "This sign is part of the GTSRB dataset and helps guide traffic flow and safety for road users."
        )
        st.markdown(f"### 🛡️ Safety Note\n{explanation}")

        # 🔹 AI (Gemini) explanation
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

        
    # Optional GPT explanation
    # gpt_text = generate_gpt_explanation(best_label)
    # if gpt_text:
    #     st.markdown("### 🤖 GPT Explanation")
    #     st.write(gpt_text)
    # else:
    #     st.markdown("### 🤖 GPT Explanation")
    #     st.write("_GPT explanation is unavailable (no API key configured or an error occurred)._")



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
