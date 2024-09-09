import cv2 
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionPredictor

# Load YOLO model for object detection
yolo_model = YOLO("yolo/best.pt")

# Define your pants model class (replace YourPantsModelClass with the actual name of your model class)
# class YourPantsModelClass(torch.nn.Module):
#     def __init__(self):
#         super(YourPantsModelClass, self).__init__()
#         # Define your model architecture here
    
#     def forward(self, x):
#         # Define the forward pass of your model
#         return x  # Example placeholder
    
# Load the pants classification model
# pants_model_state_dict = torch.load('Cnn model trouser/resnet_model.pth')
# pants_model = YourPantsModelClass()  # Instantiate your pants model class
# pants_model.load_state_dict(pants_model_state_dict)
pants_model_state_dict = torch.load('Cnn model trouser/resnet_model.pth')
pants_model = YourPantsModelClass()  # Instantiate your pants model class
pants_model.load_state_dict(pants_model_state_dict)

# Load the t-shirt classification model
tshirt_model = torch.load('Cnn model tshirt/classification_tshirt.pth')
tshirt_model.eval()

# Define transforms for preprocessing images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to classify pants
def classify_pants(image):
    input_image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = pants_model(input_image)
    probabilities = F.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    return predicted_class

# Function to classify t-shirt
def classify_tshirt(image):
    input_image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = tshirt_model(input_image)
    probabilities = F.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    return predicted_class

# Function to predict using YOLO and classify
def predict_and_classify(frame):
    results = yolo_model.predict(source=frame, show=True)
    for result in results:
        if result.object_class == "pants":
            predicted_class = classify_pants(result.region)
            print("Pants color:", "Black" if predicted_class == 0 else "Another")
        elif result.object_class == "tshirt":
            predicted_class = classify_tshirt(result.region)
            print("T-shirt color:", "White" if predicted_class == 0 else "Another")

# Capture video from camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to PIL Image for processing
    frame_pil = Image.fromarray(frame)
    
    # Predict and classify
    predict_and_classify(frame_pil)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
