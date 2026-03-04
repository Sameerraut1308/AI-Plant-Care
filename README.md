# PlantCare AI: Intelligent Plant Disease Classification System

## Table of Contents

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

## Overview

PlantCare AI is an intelligent plant disease classification system designed to provide accurate and efficient plant disease diagnosis using advanced deep learning techniques. By leveraging transfer learning with pre-trained convolutional neural networks, the system enables rapid and reliable plant pathology analysis suitable for agricultural professionals, farmers, and gardeners.

The project utilizes the New Plant Diseases Dataset, which contains over 87,000 annotated plant leaf images categorized into 38 distinct disease classes. This comprehensive dataset includes various diseases affecting major crops including Apple, Corn, Grape, Potato, Tomato, and others.

## Problem Statement

Plant diseases significantly impact agricultural productivity and crop yields worldwide. Current disease detection methods rely heavily on manual inspection by agricultural experts, which is time-consuming, labor-intensive, and often subject to human error. The lack of accessible and scalable diagnostic tools limits farmers' ability to quickly identify and respond to disease outbreaks, resulting in substantial crop losses.

## Solution Approach

PlantCare AI addresses this challenge through transfer learning, leveraging the pre-trained MobileNetV2 convolutional neural network (CNN) to achieve rapid model development and high classification accuracy. Transfer learning allows the model to benefit from pre-existing knowledge of image feature patterns, significantly enhancing performance while reducing computational requirements.

### Key Advantages of Transfer Learning

- Faster training times compared to training models from scratch
- Improved classification accuracy through leveraging pre-learned features
- Reduced computational costs and resource requirements
- Ability to work effectively with limited labeled data
- Scalability to diverse agricultural environments

## Key Features

- **High-Accuracy Classification**: 38-class disease classification across multiple crop types
- **Efficient Model**: MobileNetV2-based architecture optimized for deployment on diverse platforms
- **Real-Time Processing**: Fast inference enabling real-time disease identification
- **Scalable Architecture**: Designed for integration with various agricultural systems
- **Comprehensive Dataset**: Trained on over 87,000 annotated plant leaf images
- **Transfer Learning**: Leverages pre-trained models to maximize accuracy while minimizing training overhead

## Use Cases

### Scenario 1: Automated Agricultural Monitoring Systems

Integrating PlantCare AI into automated monitoring systems on farms revolutionizes crop health management. The system captures plant leaf images, classifies diseases in real-time, and generates detailed diagnostic reports. This automation:
- Reduces manual workload on agricultural experts
- Accelerates diagnostic processes
- Ensures consistent, high-accuracy results
- Improves crop yields and reduces disease-related losses

### Scenario 2: Mobile Applications for Home Gardeners

PlantCare AI deployed as a mobile application enables home gardeners and hobbyists to identify plant diseases independently. Users can capture photos of affected leaves and receive instant analysis, allowing:
- Timely intervention without expert consultation
- Self-service plant health diagnostics
- Improved accessibility to plant care expertise
- Healthy garden maintenance through early disease detection

### Scenario 3: Educational Tools for Agricultural Training

Integration into agricultural training platforms provides students and technicians with practical learning opportunities. The system enables:
- Interactive analysis of plant disease images
- Instant feedback on disease identification
- Hands-on experience with plant pathology
- Development of practical skills in modern agricultural diagnostics

## Dataset

**Dataset Name**: New Plant Diseases Dataset

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

## Model Architecture

The project employs **MobileNetV2**, a lightweight yet efficient convolutional neural network architecture specifically designed for resource-constrained environments.

### Architecture Highlights

- **Base Model**: Pre-trained MobileNetV2 on ImageNet dataset
- **Transfer Learning**: Fine-tuning of top layers with disease classification data
- **Input Size**: 224x224 RGB images
- **Output**: 38-class disease classification with confidence scores
- **Optimization**: Designed for deployment on mobile and edge devices

## Requirements

- Python 3.8+
- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- OpenCV

Detailed requirements are available in `requirements.txt`.

## Installation

1. Clone or download the repository:
   ```bash
   cd AI-Plant-Care
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the New Plant Diseases Dataset:
   - Obtain the dataset and place it in the `dataset/` directory
   - Ensure proper directory structure within `dataset/`

4. Verify the installation:
   ```bash
   python -c "import tensorflow; print(tensorflow.__version__)"
   ```

## Usage

### Data Preprocessing

Process and prepare the dataset:
```bash
python preprocessing/preprocess.py
```

### Model Training

Train the PlantCare AI model:
```bash
python model/train.py
```

### Model Evaluation

Evaluate model performance on test data:
```bash
python model/evaluate.py
```

### Making Predictions

Classify a plant leaf image:
```bash
python model/predict.py --image_path <path_to_leaf_image>
```

## Project Structure

```
AI-Plant-Care/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── dataset/                     # Plant diseases dataset directory
│   └── README.md               # Dataset documentation
├── preprocessing/               # Data preprocessing scripts
│   └── README.md               # Preprocessing documentation
├── model/                       # Model training and inference
│   ├── README.md               # Model documentation
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation script
│   └── predict.py              # Prediction/inference script
└── main.py                      # Main entry point
```

## Contributing

Contributions to PlantCare AI are welcome. Please ensure:
- Code follows project conventions
- Changes are documented appropriately
- Testing is performed thoroughly
- Pull requests include clear descriptions of modifications

## License and Acknowledgments

This project utilizes the New Plant Diseases Dataset and the MobileNetV2 pre-trained model architecture. All contributions and data sources are acknowledged according to their respective licenses and guidelines.
