from keras.layers import TFSMLayer
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the model 
model = TFSMLayer(
    "converted_savedmodel/model.savedmodel",  
    call_endpoint="serving_default"
)

# image
img = Image.open("test.jpg") 
img = img.resize((224, 224))  
img_array = np.expand_dims(np.array(img) / 255.0, axis=0).astype(np.float32)

# Run prediction
output = model(img_array)

# Print prediction
print("Raw model output:", output)

for key, value in output.items():
    predicted_index = np.argmax(value.numpy())
    print(f"Predicted class index: {predicted_index}")
    print(f"Probabilities: {value.numpy()}")
