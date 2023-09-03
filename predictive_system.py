import numpy as np
import pickle

# Loading the trained model
loaded_model = pickle.load(open('trained_model.sav','rb'))
input_data = (8,3,10,2,10,10,10)
#changing the input data into numpy array
id_np_array = np.asarray(input_data)
id_reshaped = id_np_array.reshape(1,-1)

prediction = loaded_model.predict(id_reshaped)
print(prediction)