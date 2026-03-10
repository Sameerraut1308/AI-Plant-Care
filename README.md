 

 

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Solution Approach](#solution-approach)
- [Key Features](#key-features)
- [Use Cases](#use-cases)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

 
PlantCare AI is a deep learning based system that detects plant diseases using leaf images. The system uses transfer learning with the MobileNetV2 model trained on the PlantVillage dataset. Users can upload a plant leaf image and the system predicts the disease along with a confidence score.

This project was developed as part of the SmartBridge Artificial Intelligence and Machine Learning course.

---

 
Plant diseases significantly reduce crop productivity and quality. Early detection of plant diseases can help farmers take preventive actions and reduce agricultural losses.

---

 
This system uses a convolutional neural network with transfer learning to classify plant diseases from leaf images. A Streamlit-based web interface allows users to upload images and receive disease predictions.

---

 
- Python
- TensorFlow / Keras
- Transfer Learning (MobileNetV2)
- Streamlit
- NumPy
- PlantVillage Dataset

--- 

 

- Faster training times compared to training models from scratch
- Improved classification accuracy through leveraging pre-learned features
- Reduced computational costs and resource requirements
- Ability to work effectively with limited labeled data
- Scalability to diverse agricultural environments

 

- **High-Accuracy Classification**: 38-class disease classification across multiple crop types
- **Efficient Model**: MobileNetV2-based architecture optimized for deployment on diverse platforms
- **Real-Time Processing**: Fast inference enabling real-time disease identification
- **Scalable Architecture**: Designed for integration with various agricultural systems
- **Comprehensive Dataset**: Trained on over 87,000 annotated plant leaf images
- **Transfer Learning**: Leverages pre-trained models to maximize accuracy while minimizing training overhead

 

 

Integrating PlantCare AI into automated monitoring systems on farms revolutionizes crop health management. The system captures plant leaf images, classifies diseases in real-time, and generates detailed diagnostic reports. This automation:
- Reduces manual workload on agricultural experts
- Accelerates diagnostic processes
- Ensures consistent, high-accuracy results
- Improves crop yields and reduces disease-related losses

 

PlantCare AI deployed as a mobile application enables home gardeners and hobbyists to identify plant diseases independently. Users can capture photos of affected leaves and receive instant analysis, allowing:
- Timely intervention without expert consultation
- Self-service plant health diagnostics
- Improved accessibility to plant care expertise
- Healthy garden maintenance through early disease detection

 

Integration into agricultural training platforms provides students and technicians with practical learning opportunities. The system enables:
- Interactive analysis of plant disease images
- Instant feedback on disease identification
- Hands-on experience with plant pathology
- Development of practical skills in modern agricultural diagnostics

 

**Dataset Name**: New Plant Diseases Dataset

Dataset source:
https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

The dataset contains images of plant leaves belonging to multiple plant species and disease categories.

Due to GitHub storage limitations, the dataset is not included in this repository.

**Dataset Statistics**:
- Total Images: 87,000+
- Number of Classes: 38
- Categories: Plant diseases across major crops
- Crops Covered: Apple, Corn, Grape, Potato, Tomato, and additional species
- Annotation Quality: Comprehensive manual annotation

**Data Characteristics**:
- Diverse environmental conditions
- Varied lighting and angles
- High-resolution leaf images
- Balanced representation across disease categories

- ## Project Workflow

 
The PlantVillage dataset containing plant leaf images of multiple diseases, was used.

 
Images were resized to 224x224 and normalized before feeding them into the model.

 
Transfer learning using MobileNetV2 was used to train the plant disease detection model.

 
The model was trained on the dataset to classify plant diseases into multiple classes.

 
The trained model predicts the plant disease and displays a confidence score.

 
A Streamlit web application was developed to allow users to upload plant leaf images and receive predictions.

---


 

1. Install required libraries

pip install -r requirements.txt

2. Run the application

streamlit run app.py

---


 

 

Process and prepare the dataset:
```bash
python preprocessing/preprocess.py
```

 

Train the PlantCare AI model:
```bash
python model/train.py
```

 

Evaluate model performance on test data:
```bash
python model/evaluate.py
```

 

Classify a plant leaf image:
```bash
python model/predict.py --image_path <path_to_leaf_image>
```

 
The system predicts:

- Plant type
- Disease name
- Confidence score

---

 
- Add remedy suggestions for detected diseases
- Deploy the system as a web application
- Improve model accuracy with additional training data

 
Contributions to PlantCare AI are welcome. Please ensure:
- Code follows project conventions
- Changes are documented appropriately
- Testing is performed thoroughly
- Pull requests include clear descriptions of modifications

 

This project utilizes the New Plant Diseases Dataset and the MobileNetV2 pre-trained model architecture. All contributions and data sources are acknowledged according to their respective licenses and guidelines.
