import numpy as np
import cv2
import streamlit as st
from PIL import Image

st.title("Black and White to Color Image Convertor")
file = st.file_uploader("Upload Your Image", type=['jpeg','jpg', 'bmp', 'png'])
    
if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    img = np.array(image)
    
    st.text("Your original image")
    st.image(image, use_column_width=True)
    st.text("Your colorized image")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    net = cv2.dnn.readNetFromCaffe('colorization_deploy_v2.prototxt','colorization_release_v2.caffemodel')
    pts = np.load('pts_in_hull.npy')

    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)

    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    scaled = img.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (img.shape[1], img.shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
    colorized = np.clip(colorized, 0, 1)
  
    colorized = (255 * colorized).astype("uint8")    
    
    st.image(colorized, use_column_width=True)