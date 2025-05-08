import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

# Title
st.title("Brain MRI Tumour Detection")
st.write("Upload a brain MRI scan to detect if it shows a tumour.")

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = models.resnet18(pretrained=False)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2)
        )
        model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        raise

model = load_model()
classes = ['No Tumour', 'Tumour']

# Define image transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded MRI Scan", use_container_width=True)


    # Preprocess and predict
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence = probs[0].max().item() * 100
        prediction = classes[probs[0].argmax().item()]

    st.write(f"### Prediction: {prediction}")
    st.write(f"**Confidence:** {confidence:.2f}%")
