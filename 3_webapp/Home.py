import streamlit as st
from predictions import YOLO_Pred  # Ensure this module is available and correctly implemented
from PIL import Image
import numpy as np

st.set_page_config(page_title="YOLO Object Detection", layout='wide')

# Header
st.title('YOLO Object Detection')

with st.spinner('Loading YOLO model...'):
    yolo = YOLO_Pred(onnx_model='./models/best.onnx', data_yaml='./models/data.yaml')

def upload_image():
    # Upload Image
    image_file = st.file_uploader(label='Upload Image', type=['png', 'jpg', 'jpeg'])
    if image_file is not None:
        size_mb = image_file.size / (1024 ** 2)
        file_details = {
            "filename": image_file.name,
            "filetype": image_file.type,
            "filesize": f"{size_mb:.2f} MB"
        }
        st.success('VALID IMAGE file type (png or jpeg)')
        return {"file": image_file, "details": file_details}
    else:
        st.warning('Please upload a valid image file (png, jpg, jpeg).')
        return None

def main():
    st.sidebar.header("Image Upload")
    uploaded_object = upload_image()
    
    if uploaded_object:
        image_obj = Image.open(uploaded_object['file'])
        
        # Displaying image and details
        st.sidebar.subheader('File Details')
        st.sidebar.json(uploaded_object['details'])
        
        st.image(image_obj, caption='Uploaded Image', use_column_width=True)
        
        if st.sidebar.button('Get Detection from YOLO'):
            with st.spinner('Processing image...'):
                image_array = np.array(image_obj)
                pred_img = yolo.predictions(image_array)
                pred_img_obj = Image.fromarray(pred_img)
                
                st.image(pred_img_obj, caption='Detected Objects', use_column_width=True)
                st.success('Detection complete!')

if __name__ == "__main__":
    main()
