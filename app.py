# -*- coding: utf-8 -*-
"""

@author: jibin
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import math
from streamlit_lottie import st_lottie
import requests
from streamlit_lottie import st_lottie_spinner
import time
import json
import joblib
import cv2
import pywt
import plotly.graph_objects as go
from PIL import Image


#adding animations using lottie
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
    
    
lottie_welcome = load_lottiefile("lottie/welcome.json")
lottie_home = load_lottiefile("lottie/home.json")
lottie_coding = load_lottiefile("lottie/coding.json")
lottie_download = load_lottiefile("lottie/download.json")

#change default page name and icon
st.set_page_config(page_title="Max Machine Learning Projects", page_icon=":dragon:", layout="wide")

#minimize default features(footer)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: visible;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# loading the saved models



# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Main Menu',

                          ['Home Page',
                           'BH Price Prediction',
                           'Facial Recognition'],
                          icons=['house','person'],
                          default_index=0, menu_icon="menu-button-wide-fill")
    
    
 #Home Page Configuration   
if (selected == 'Home Page'):
    st_lottie(lottie_welcome, loop=True, key='home', height=430)
    # Use local CSS
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    local_css("style/style.css")

    
    # ---- HEADER SECTION ----
    with st.container():
        st.subheader("Hi, I am Jibin :wave:")
        st.title("A Data Analyst Intern From India:earth_asia:")
        st.write("I am a **Machine Learning** enthusiast and a **Python Developer** with a **Bachelor of Technology** focused in **Electrical and Electronics Engineering** from Vimal Jyothi Engineering College Kannur. I have 1.5 years of experience in the Engineering Sector. My activities are much beyond my stream of education. Currently, I am passionate about finding ways to use **Python & Machine Learning** to be more efficient and effective in a business setting"
                     
                           )
        st.write("[Learn More :on: >](https://drive.google.com/file/d/1SsshlyA38LdDw9wLZQdxXwHHvfrJhxr5/view?usp=sharing)")
      
    # ---- WHAT I DO ----
    with st.container():
        st.write("---")
        left_column, right_column = st.columns(2)
        with left_column:
            st.header(":male-student:What do I do and What have I done?")
            st.write("##")
            st.write(
                """
                - :male-student:Current  **Artificial intelligence Data Scientist**  Intern at **NASSCOM**(DDU-GKY) 
                - :100:Iam looking for a way to leverage the power of **Python** in their day-to-day work.
                - :sparkles:Try to learn **Data Analysis** & **Data Science** to perform meaningful and impactful analyses.
                - :footprints:Learn to identify appropriate quantitative methods to analyze datasets and to address, visualize and predict data.
                - :rainbow:Learn **Machine Learning Projects** from Various sources like YouTube
                - Travel:steam_locomotive:, Eat:poultry_leg:, Sleep:sleeping:
                
                """
                )
           
    with right_column:
        st_lottie(lottie_coding, height=400, key="coding")

  # ---- connect me on ----
    with st.container():
       st.write("---")
       st.header(":link:Connect Me On")
       st.write("##")  
    
    lottie_linke = load_lottieurl("https://assets8.lottiefiles.com/private_files/lf30_tgzwnxcf.json")

    contact_icon, contact_link = st.columns((0.05, 1))
    with contact_icon:
        st_lottie(lottie_linke, height=60, key="linke")
    with contact_link:
        st.subheader("[**1. Linkedin**](https://www.linkedin.com/in/jibin-tom-2b04551b7/)")
 
    
    lottie_git = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_dK9q4K.json")

    contact_icon, contact_link = st.columns((0.05, 1))
    with contact_icon:
        st_lottie(lottie_git, height=60, key="git")
    with contact_link:
        st.subheader("[**2. GitHub**](https://github.com/jibintom)")
        
    lottie_mail = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_u25cckyh.json")

    contact_icon, contact_link = st.columns((0.05, 1))
    with contact_icon:
        st_lottie(lottie_mail, height=60, key="mail")
    with contact_link:
        st.subheader("[**3. Email**](jibintom1997@gmail.com)")

    # ---- CONTACT ----
    with st.container():
        st.write("---")
        st.header(":mailbox_closed: Get In Touch With Me!")
        st.write("##")

    # Documention: https://formsubmit.co/ !!! CHANGE EMAIL ADDRESS !!!
    contact_form = """
    <form action="https://formsubmit.co/jibintom3@gmail.com" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your name" required>
        <input type="email" name="email" placeholder="Your email" required>
        <textarea name="message" placeholder="Your message here" required></textarea>
        <button type="submit">Send</button>
    </form>
    """
    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown(contact_form, unsafe_allow_html=True)
    with right_column:
        st.empty()
 

# BHP Prediction Page 

if (selected == 'BH Price Prediction'):
    
    # pagre title
    st.title(':house: Bangalore House Price Prediction using ML')
    st_lottie(lottie_home, key="home", height=400, loop=True)

    # getting the input data from the user
    location_list=['','Whitefield', 'Sarjapur  Road', 'Electronic City', 'Kanakpura Road', 'Thanisandra', 'Yelahanka', 'Uttarahalli', 'Hebbal', 'Marathahalli', 'Raja Rajeshwari Nagar', 'Bannerghatta Road', 'Hennur Road', '7th Phase JP Nagar', 'Haralur Road', 'Electronic City Phase II', 'Rajaji Nagar', 'Chandapura', 'Bellandur', 'Hoodi', 'KR Puram', 'Electronics City Phase 1', 'Yeshwanthpur', 'Begur Road', 'Sarjapur', 'Kasavanhalli', 'Harlur', 'Hormavu', 'Banashankari', 'Kengeri', 'Ramamurthy Nagar', 'Koramangala', 'Varthur', 'Old Madras Road', 'Hosa Road', 'Jakkur', 'JP Nagar', 'Kothanur', 'Kaggadasapura', 'Nagarbhavi', 'Akshaya Nagar', 'Thigalarapalya', 'TC Palaya', 'Rachenahalli', 'Malleshwaram', '8th Phase JP Nagar', 'Budigere', 'HSR Layout', 'Jalahalli', 'Hennur', 'Panathur', 'Bisuvanahalli', 'Jigani', 'Hulimavu', 'Ramagondanahalli', 'Hegde Nagar', 'Bhoganhalli', 'Gottigere', 'Mysore Road', 'Kundalahalli', 'Brookefield', 'Hosur Road', 'Balagere', 'Indira Nagar', 'Vidyaranyapura', 'Subramanyapura', 'Vittasandra', 'CV Raman Nagar', '9th Phase JP Nagar', 'Kadugodi', 'Kanakapura', 'Vijayanagar', 'Attibele', 'Horamavu Agara', 'Talaghattapura', 'Devanahalli', 'Yelahanka New Town', 'Kengeri Satellite Town', '5th Phase JP Nagar', 'Green Glen Layout', 'Sahakara Nagar', 'Channasandra', 'Kudlu Gate', 'Bommasandra', 'Lakshminarayana Pura', 'Hosakerehalli', 'Anekal', 'R.T. Nagar', 'Frazer Town', 'Hebbal Kempapura', 'Bommanahalli', 'Kalena Agrahara', 'Tumkur Road', 'Old Airport Road', 'Basavangudi', 'Ambedkar Nagar', 'Mahadevpura', 'Doddathoguru', 'Chikkalasandra', 'Ananth Nagar', 'Kumaraswami Layout', 'Dodda Nekkundi', 'Kudlu', 'Kammasandra', 'BTM 2nd Stage', 'Padmanabhanagar', 'Somasundara Palya', 'Horamavu Banaswadi', 'Banashankari Stage III', 'Singasandra', 'Kodichikkanahalli', 'Ambalipura', 'Anandapura', 'Choodasandra', 'Kothannur', 'Margondanahalli', 'Bommasandra Industrial Area', 'Gubbalala', 'Babusapalaya', 'Kogilu', 'Magadi Road', 'Seegehalli', 'Iblur Village', 'Munnekollal', 'Battarahalli', '1st Phase JP Nagar', 'Abbigere', 'Amruthahalli', 'Ardendale', '2nd Stage Nagarbhavi', 'Kambipura', 'Lingadheeranahalli', '6th Phase JP Nagar', 'EPIP Zone', 'Thubarahalli', 'Kathriguppe', 'Kaval Byrasandra', 'Rayasandra', 'Domlur', 'Sonnenahalli', 'Gunjur', 'Hoskote', 'Ulsoor', 'Basaveshwara Nagar', 'Poorna Pragna Layout', 'HBR Layout', 'Binny Pete', 'Sanjay nagar', 'Yelachenahalli', 'OMBR Layout', 'Kalyan nagar', 'HRBR Layout', 'Pai Layout', 'Kaggalipura', 'Kannamangala', 'Garudachar Palya', 'Billekahalli', 'Devarachikkanahalli', 'Chikka Tirupathi', 'Banaswadi', 'Dasarahalli', 'Kammanahalli', 'Kaikondrahalli', 'Bannerghatta', 'Sarjapura - Attibele Road', 'Malleshpalya', 'Nagavara', 'Begur', 'Anjanapura', 'Kasturi Nagar', 'Mallasandra', 'Banashankari Stage II', 'Parappana Agrahara', 'Kereguddadahalli', 'BTM Layout', 'Kenchenahalli', 'Arekere', 'Sector 2 HSR Layout', 'Dasanapura', 'Cooke Town', 'Banashankari Stage VI', 'Nagavarapalya', 'Kodihalli', 'Judicial Layout', 'Varthur Road', 'Benson Town', 'Sultan Palaya', 'Jalahalli East', 'NGR Layout', 'Chamrajpet', 'Doddakallasandra', 'NRI Layout', 'Murugeshpalya', '1st Block Jayanagar', 'Sarakki Nagar', 'Giri Nagar', 'Konanakunte', 'Vishveshwarya Layout', 'Neeladri Nagar', 'Shampura', 'Gollarapalya Hosahalli', 'Yelenahalli', 'Kodigehaali', 'Prithvi Layout', 'Kadubeesanahalli', 'BEML Layout', 'ISRO Layout', 'Chikkabanavar', 'Rajiv Nagar', 'Mahalakshmi Layout', 'Dommasandra', 'Cunningham Road', 'Sector 7 HSR Layout', 'Shivaji Nagar', 'Badavala Nagar', '5th Block Hbr Layout', 'Mico Layout', 'ITPL', 'Nagasandra', 'Banashankari Stage V', 'Vishwapriya Layout', 'Bharathi Nagar', 'Sompura', 'Karuna Nagar', 'Vasanthapura', 'Cox Town', 'GM Palaya', 'AECS Layout', 'Laggere', 'Pattandur Agrahara', 'Doddaballapur', 'Tindlu', 'Marsur', 'Bommenahalli', 'Narayanapura', 'Nehru Nagar', 'HAL 2nd Stage', 'Banjara Layout', 'LB Shastri Nagar', 'Kodigehalli', '2nd Phase Judicial Layout', 'Others']
    location=st.selectbox('Select Your Location',location_list)
    sqft=st.slider('Select Total_sqft', 350, 5000)
    bath=st.slider('Select Number of Bathroom', 1, 6)
    bhk=st.slider('Select Required BHK', 1, 8)
    
    # code for Prediction
    df=pd.read_csv("https://raw.githubusercontent.com/jibintom/Machine-Learning-Deployment-Using-Streamlit/main/bhp_final_data_for_training.csv")
    x = df.drop(["price"],axis='columns')
    y = df.price
    
    
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=10)
    
    bhp_model = LinearRegression()
    bhp_model.fit(x_train,y_train)
    bhp_prediction = ''
    bhp_result = ''
    
    # creating a button for Prediction
    
    if st.button('Predict Home Price'):
        with st_lottie_spinner(lottie_download, key="download", height=60):
            time.sleep(2)
        st.balloons()
        
        def predict_price(location,sqft,bath,bhk):    
            loc_index = np.where(x.columns==location)[0][0]
            j = np.zeros(len(x.columns))
            j[0] = sqft
            j[1] = bath
            j[2] = bhk
            if loc_index >= 0:
                j[loc_index] = 1

            return bhp_model.predict([j])[0]
        bhp_prediction = predict_price(location, sqft, bath, bhk)
        bhp_prediction=math.trunc(bhp_prediction)
     
        if (bhp_prediction <100):
          bhp_result ="Price is  ",bhp_prediction,"  Lakhs"
        else:
          cr=bhp_prediction/100
          bhp_result ='Price is  ' , cr , "Crores"
        
    st.success(bhp_result)
  



#FR Page Configuration   

if (selected == 'Facial Recognition'):
   
    # page title
    st.title(":two_women_holding_hands:Facial Recognition using ML")




#wavlet

def w2d(img, mode='haar', level=1):
    imArray = img
    #Datatype conversions
    #convert to grayscale
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    #convert to float
    imArray =  np.float32(imArray)
    imArray /= 255;
    # compute coefficients
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)
    coeffs_H[0] *= 0;

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H =  np.uint8(imArray_H)

    return imArray_H



#predict function
__class_name_to_number = {}
__class_number_to_name = {}

__model = None

def classify_image(our_image, file_path=None):
    result = []
    result_img=[]
    person=[]
    score=[]
    imgs = get_cropped_image_if_2_eyes(file_path, our_image)
    if (len(imgs)==0):
       result="The eyes and face are not properly visible in the given image. Try with another one."
       result_img=our_image
       person=None
       score=None
       return result,result_img,person,score

    else:
        for img in imgs:
            scalled_raw_img = cv2.resize(img, (32, 32))
            img_har = w2d(img, 'db1', 5)
            scalled_img_har = cv2.resize(img_har, (32, 32))
            combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))
        
            len_image_array = 32*32*3 + 32*32
        
            final = combined_img.reshape(1,len_image_array).astype(float)
            
            
            person=class_number_to_name(__model.predict(final)[0])
            score=max(np.around(__model.predict_proba(final)*100,2).tolist()[0])
            result.append({
                'Person': class_number_to_name(__model.predict(final)[0]),
                'Probability Score': max(np.around(__model.predict_proba(final)*100,2).tolist()[0])
                
            })
            img2 = np.array(our_image.convert('RGB'))
            gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        # Draw rectangle around the faces
            name='Unknown'
            for (x, y, w, h) in faces:
          # To draw a rectangle in a face
              cv2.rectangle(img2, (x, y), (x + w, y + h), (255, 255, 0), 2)
              
            if (score> 48.9):
                  name = class_number_to_name(__model.predict(final)[0])
                  result_img=cv2.putText(img2, name, (x, y + h), cv2.FONT_HERSHEY_COMPLEX_SMALL,3.0,(0, 0, 255),3 )
                  #status=final_result,"with a probality of ", score
                  #final
                  #score
                  
            elif (score<48.9):
                  result_img=cv2.putText(img2, 'Unknown', (x, y + h), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3.0, (0, 0, 255),3)
                  
            
        
        return result,result_img,person,score
    
    

def class_number_to_name(class_num):
    return __class_number_to_name[class_num]

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name

    with open(r'./Face Recognition/artifacts/class_dictionary.json' , "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}

    global __model
    if __model is None:
        with open(r'./Face Recognition/artifacts/saved_model.pkl', 'rb') as f:
            __model = joblib.load(f)
            
    print("loading saved artifacts...done")


def get_cv2_image_from_base64_string(our_image):
  
    img = np.array(our_image.convert('RGB'))
    return img

def get_cropped_image_if_2_eyes(image_path, our_image):
    global face_cascade
    global eye_cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'Face Recognition/opencv/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'Face Recognition/opencv/haarcascades/haarcascade_eye.xml')

    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(our_image)
    global gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    global faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    

    cropped_faces = []
    for (x,y,w,h) in faces:
            global roi_gray
            roi_gray = gray[y:y+h, x:x+w]
            global roi_color
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) >= 2:
                cropped_faces.append(roi_color)
    return cropped_faces





if __name__ == '__main__':
    load_saved_artifacts()


   
def main():
    """Face Recognition App"""
    st.write(" ")

    html_temp = """
    <body style="background-color:red;">
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Face Recognition WebApp</h2>
    </div>
    </body>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if image_file is not None:
       our_image = Image.open(image_file)
       
    if st.button("Recognise"):
        result,result_img,person,score= classify_image(our_image)
        st.title("Image")
        st.image(result_img)
        
        
        st.title("Result")
        if person==None:
            st.subheader(result)
                
        elif score> 48.9:
            
            df=pd.DataFrame(result)
            st.header(person)
            fig = go.Figure(data=[go.Table(
                header=dict(values=list(df.columns),
                            line_color='darkslategray',
                            fill_color='steelblue',
                            align='left'),
                cells=dict(values=[df.Person,df["Probability Score"]],
                           line_color='darkmagenta',
                           fill_color='tomato',
                           align='left'))
                ])
            fig.update_layout(width=1000, height=800)
            st.write(fig)
        else:
            
            return st.subheader("The given image is not available in the data set. Try a new one")
   
    

if __name__ == '__main__':
    main()
    
    


