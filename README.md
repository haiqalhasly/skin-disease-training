# Skin Disease Detection using Machine Learning 🧑‍⚕️

## Using YOLO, Roboflow and Python 💻

This project is my learning and building process to train datasets from Roboflow with yolov8n model.🤖 It is a machine learning model that will help ease your process. I use OpenCV to show real time detection of my model
Here's what I have achieved :

  * ✅ Finding 5 good datasets from Roboflow and how to implement in training
  * ✅ Train machine learning model using 1000+ image datasets to achieve accuracy
  * ✅ How to setup environment for machine learning in VSCode

## Model Output Example 📷

Below is a sample prediction

<img src ="images/acne.png" width="400"/>
<img src ="images/basalCellCarcinoma.png" width="400"/>
<img src ="images/eczema.png" width="400"/>
<img src ="images/virtiligo.png" width="400"/>
<img src ="images/warts.png" width="400"/>

How to use ? 📖

 1. Clone this project
 2. Find dataset from [Roboflow](https://universe.roboflow.com) and download to your computer ( make sure it's already divide to train,test and eval)
 3. Add the dataset to your project folder and IDE
 4. Open training.ipynb and run the 1st cell
 5. You can use 2nd cell if the dataset is not already divided to train, test and eval
 6. For the 3rd cell, find data.yaml in your dataset folder and copy the path. ( make sure to change from \ to this / for the path)
 7. Run the cell and let your machine learning
 8. There will be runs or train folder appear and find weights within it. best.pt will be your AI model.

Thank youu!🥰
