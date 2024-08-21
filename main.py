import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import argparse
import ssl

ssl._create_default_https_context = ssl._create_stdlib_context

CLASS_NAMES = ["High squamous intra-epithelial lesion","Low squamous intra-epithelial lesion",
           "Negative for Intraepithelial malignancy","/content/dataset/Squamous cell carcinoma"]
MODEL_NAME = "cancer_best_model.pth"


def predict(model_name, img_path):
    # load the model
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(in_features=256, out_features=4, bias=True)

    weights = torch.load(model_name,map_location ='cpu')
    model.load_state_dict(weights)

    # preprocess the image
    prep_img_mean = [0.485, 0.456, 0.406]
    prep_img_std = [0.229, 0.224, 0.225]
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=prep_img_mean, std=prep_img_std),
        ]
    )
    image = Image.open(img_path)
    preprocessed_image = transform(image).unsqueeze(0)

    # predict the class
    model.eval()
    output = model(preprocessed_image)
    pred_idx = torch.argmax(output, dim=1)
    predicted_class = CLASS_NAMES[pred_idx]
    return predicted_class


def create_app(model_name):
    # title
    st.title("Cervical Cancer Classification App")

    # file uploader
    uploaded_file = st.file_uploader(
        "Choose an image to classify", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        st.write("")

        # predict the class
        predicted_class = predict(model_name, uploaded_file)

        col_left, col_right = st.columns(2)

        # the Predict button with the predicted class
        with col_left:
            if st.button("Predict"):
                st.markdown(f"## {predicted_class} ")

        # display the image
        with col_right:
            image = Image.open(uploaded_file)
            st.image(image)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, help="model name", default=MODEL_NAME
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    create_app(model_name=args.model)