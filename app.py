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



#adding animations using lottie
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url_welcome = "https://assets10.lottiefiles.com/packages/lf20_puciaact.json"
lottie_welcome = load_lottieurl(lottie_url_welcome)
lottie_url_home = "https://assets6.lottiefiles.com/private_files/lf30_p5tali1o.json"
lottie_home = load_lottieurl(lottie_url_home)
lottie_url_diabetic = "https://assets4.lottiefiles.com/packages/lf20_tbjuenb2.json"
lottie_diabetic = load_lottieurl(lottie_url_diabetic)
lottie_url_heart = "https://assets2.lottiefiles.com/packages/lf20_YO8gSl.json"
lottie_heart = load_lottieurl(lottie_url_heart)
lottie_url_parkinson = "https://assets7.lottiefiles.com/packages/lf20_gkgqj2yq.json"
lottie_parkinson = load_lottieurl(lottie_url_parkinson)
lottie_coding = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_w51pcehl.json")
lottie_url_download = "https://assets9.lottiefiles.com/packages/lf20_YSMTHj.json"
lottie_download = load_lottieurl(lottie_url_download)

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
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))

heart_disease_model = pickle.load(open('heart_disease_model.sav','rb'))

parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))


# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Main Menu',

                          ['Home Page',
                           'BH Price Prediction',
                           'Diabetes Prediction',
                           'Heart Disease Prediction',
                           'Parkinsons Prediction'],
                          icons=['house','house-door-fill','activity','heart','person'],
                          default_index=0, menu_icon="menu-button-wide-fill")
    
    
 #Home Page Configuration   
if (selected == 'Home Page'):
    st_lottie(lottie_welcome, loop=True, key='home', height=430)
    # Use local CSS
    def local_css(file_name):
        with open("https://raw.githubusercontent.com/jibintom/Machine-Learning-Deployment-Using-Streamlit/main/style/style.css") as f:
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
    df=pd.read_csv("F:/DS Files/BHP ML Prediction/bhp_final_data_for_training.csv")
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
  
  
# Diabetes Prediction Page
if (selected == 'Diabetes Prediction'):
    
    # page title
    st.title(':syringe:Diabetes Prediction using ML')
    
    st_lottie(lottie_diabetic, key="diabetic", loop=True, height=400)
    
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        
    with col2:
        Glucose = st.text_input('Glucose Level')
    
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    
    with col2:
        Insulin = st.text_input('Insulin Level')
    
    with col3:
        BMI = st.text_input('BMI value')
    
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    
    with col2:
        Age = st.text_input('Age of the Person')
    
    
    # code for Prediction
    diab_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        with st_lottie_spinner(lottie_download, key="download", height=60):
            time.sleep(2)
        st.balloons()
        
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        if (diab_prediction[0] == 1):
          diab_diagnosis = 'The person is diabetic'
        else:
          diab_diagnosis = 'The person is not diabetic'
        
    st.success(diab_diagnosis)




# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):
    
    # page title
    st.title(':broken_heart:Heart Disease Prediction using ML')
    
    st_lottie(lottie_heart, key="heart", loop=True, height=450)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
        
    with col2:
        sex = st.text_input('Sex')
        
    with col3:
        cp = st.text_input('Chest Pain types')
        
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
        
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
        
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
        
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
        
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
        
    with col3:
        exang = st.text_input('Exercise Induced Angina')
        
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
        
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
        
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
        
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
        
        
     
     
    # code for Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Heart Disease Test Result'):
        heart_prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]])                          
        
        if (heart_prediction[0] == 1):
          heart_diagnosis = 'The person is having heart disease'
        else:
          heart_diagnosis = 'The person does not have any heart disease'
        
    st.success(heart_diagnosis)
        
    
    

# Parkinson's Prediction Page
if (selected == "Parkinsons Prediction"):
    
    # page title
    st.title(":two_women_holding_hands:Parkinson's Disease Prediction using ML")
    
    st_lottie(lottie_parkinson, key="parkinson", height=450, loop=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
        
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
        
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
        
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
        
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        
    with col1:
        RAP = st.text_input('MDVP:RAP')
        
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
        
    with col3:
        DDP = st.text_input('Jitter:DDP')
        
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
        
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
        
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
        
    with col3:
        APQ = st.text_input('MDVP:APQ')
        
    with col4:
        DDA = st.text_input('Shimmer:DDA')
        
    with col5:
        NHR = st.text_input('NHR')
        
    with col1:
        HNR = st.text_input('HNR')
        
    with col2:
        RPDE = st.text_input('RPDE')
        
    with col3:
        DFA = st.text_input('DFA')
        
    with col4:
        spread1 = st.text_input('spread1')
        
    with col5:
        spread2 = st.text_input('spread2')
        
    with col1:
        D2 = st.text_input('D2')
        
    with col2:
        PPE = st.text_input('PPE')
        
    
    
    # code for Prediction
    parkinsons_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])                          
        
        if (parkinsons_prediction[0] == 1):
          parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
          parkinsons_diagnosis = "The person does not have Parkinson's disease"

        
    st.success(parkinsons_diagnosis)



