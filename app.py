import streamlit as st
import numpy as np
import cv2
import os, urllib.request

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

PROTOTXT = os.path.join(MODEL_DIR, "colorization_deploy_v2.prototxt")
CAFFEMODEL = os.path.join(MODEL_DIR, "colorization_release_v2.caffemodel")
PTS_IN_HULL = os.path.join(MODEL_DIR, "pts_in_hull.npy")

# --- Download caffe model from Google Drive if missing ---
if not os.path.exists(CAFFEMODEL):
    st.info("Downloading pretrained model (~123MB)... please wait")
    # Replace with your actual Google Drive direct download link
    url = "https://drive.google.com/uc?export=download&id=1-pkEOZKs78Iq5RUJEswONaLIgfBhx-78"
    urllib.request.urlretrieve(url, CAFFEMODEL)

# --- Load pretrained model ---
net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)
pts = np.load(PTS_IN_HULL)

# Add cluster centers to the model
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
net.getLayer(class8).blobs = [pts.transpose().reshape(2, 313, 1, 1)]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# --- Streamlit UI ---
st.title("🎨 Black & White Image Colorization (Pretrained OpenCV Model)")
st.write("Upload a grayscale photo and see accurate colorization!")

uploaded_file = st.file_uploader("Upload a grayscale image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    gray_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    st.image(gray_img, caption="Original Grayscale", use_column_width=True)

    # Convert grayscale to 3-channel RGB
    img_rgb = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
    l_channel = img_lab[:, :, 0]

    # Prepare input for network
    net.setInput(cv2.dnn.blobFromImage(l_channel, 1.0, (224, 224), (50,), swapRB=False))
    ab_channels = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab_channels = cv2.resize(ab_channels, (gray_img.shape[1], gray_img.shape[0]))

    # Combine L + ab channels
    colorized_lab = np.concatenate((l_channel[:, :, np.newaxis], ab_channels), axis=2)
    colorized_bgr = cv2.cvtColor(colorized_lab.astype("uint8"), cv2.COLOR_Lab2BGR)

    st.image(colorized_bgr, caption="Colorized Output", use_column_width=True)
