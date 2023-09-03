import numpy as np
import pickle
import streamlit as st  

loaded_model = pickle.load(open('trained_model.sav','rb'))
def stresslevel_prediction(input_data):
    
    #changing the input data into numpy array
    id_np_array = np.asarray(input_data)
    id_reshaped = id_np_array.reshape(1,-1)

    prediction = loaded_model.predict(id_reshaped)
    print(prediction)

    return prediction

st.markdown(
    """
    <style>
    /* Add your custom CSS here */
    body {
        
        background-image: url("https://images.pexels.com/photos/2128249/pexels-photo-2128249.jpeg?auto=compress&cs=tinysrgb&w=600");
        font-family: Arial, sans-serif; /* Font family */
    }
    .stApp {
        # background-color: ;
        max-width: 600px; /* Set the maximum width of the app */
        margin: 0 auto; /* Center the app on the page */
    }
    .stButton > button {
        background-color: #007BFF; /* Button background color */
        color: #fff; /* Button text color */
    }
    </style>
    """,
    unsafe_allow_html=True
)
    
def main():
    st.title('Are You Stressed?')
    
    Q1 = st.text_input('Q1-How many events have you Volunteered in ?')
    Q2 = st.text_input('Q2-How many events have you Participated in ?')
    Q3 = st.text_input('Q3-How many activities are you Interested in ?')
    Q4 = st.text_input('Q4-How many activities are you Passionate about ?')
    Q5 = st.text_input('Q5-How Satisfied You are with your Student Life ?')
    Q6 = st.text_input('Q6-How much effort do you make to interact with others ?')
    Q7 = st.text_input('Q7-About How events are you aware about ?')

    # Prediction code
    diagnosis = ''
    
    if st.button('PREDICT'):
        diagnosis = stresslevel_prediction([Q1,Q2,Q3,Q4,Q5,Q6,Q7])
        
    st.success(diagnosis)
    
if __name__=='__main__':
    main()