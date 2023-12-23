import numpy as np
import cv2
import streamlit as st
from streamlit_option_menu import option_menu
from tensorflow import keras
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode

# load model
emotion_dict = {0:'angry', 1 :'happy', 2: 'neutral', 3:'sad', 4: 'surprise'}
# load json and create model
json_file = open('emotion_model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)

# load weights into new model
classifier.load_weights("emotion_model1.h5")

#load face
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class Faceemotion(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        #image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)
            label_position = (x, y)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

def main():
    # Face Analysis Application #
    st.title("Pendeteksi Ekspresi Wajah")
    # horizontal menu
    selected = option_menu(None, ["Home", "Detector", "About"], 
        icons=['house', 'camera', "info-circle"], 
        menu_icon="cast", default_index=0, orientation="horizontal")
    
    if selected == "Home":
        st.subheader("Terimakasih karena sudah berkunjung!")
        html_temp_home1 = """
            <style>
                .custom-container {
                    background-color: rgba(80, 80, 80, 0.5); /* Warna abu-abu dengan opacity 0.7 */
                    padding: 10px;
                    border-radius: 10px; /* Mengatur ujung agar menjadi tumpul */
                }
                .custom-text {
                    color: white;
                    text-align: center;
                }
            </style>
            <div class="custom-container">
                <h4 class="custom-text">
                    Selamat datang di Aplikasi Pendeteksi Ekspresi Wajah!
                </h4>
                <h5 class="custom-text">
                    Aplikasi ini dapat mentracking dan mendeteksi ekspresi secara real-time.<br>
                    Pergi ke halaman "Detector" untuk mendeteksi ekspresimu. <br>
                    Jangan lupa siapkan ekspresi terbaikmu ya!
                </h5>
            </div>
        """
        st.markdown(html_temp_home1, unsafe_allow_html=True)

    elif selected == "Detector":
        st.header("Web-cam Live Feed")
        webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                        video_processor_factory=Faceemotion)
        st.subheader("Step by step")
        st.write("""
                 Berikut langkah-langkah yang harus diikuti untuk menggunakan aplikasi : 

                 1. Mengizinkan akses kamera.
                 2. Klik *SELECT DEVICE* untuk memilih device web-cam apa yang ingin digunakan.
                 3. Klik *DONE* setelah memilih device.
                 4. Klik *START* untuk mengaktifkan web-cam.
                 5. Tunggu beberapa saat sampai web-cam ditampilkan.
                 6. Setelah web-cam tampil, siapkan ekspresi terbaikmu dan biarkan aplikasi mendeteksinya!

                 """)

    elif selected == "About":
        st.subheader("Model dan Algoritma")
        st.write("""
                 Aplikasi ini dibangun menggunakan : 

                 1. OpenCV

                 2. Convolutional Neural Networks (CNN)

                 3. Streamlit

                 """)
        st.subheader("Fungsi dan Tujuan")
        st.write("""
                 Aplikasi ini memiliki fungsi dan tujuan : 

                 1. Mendeteksi wajah secara real-time melalui web-cam

                 2. Mengetahui ekspresi yang diciptakan wajah

                 3. Memiliki fitur tracking wajah

                 """)

    else:
        pass


if __name__ == "__main__":
    main()
