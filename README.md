# Loan Approval Prediction Web App 🏦🤖

This project is a **web application** built with **Flask** that uses a trained machine learning model to predict whether a loan application will be **Approved** or **Rejected** based on several input features.

The backend uses a **Random Forest Classifier** model trained on loan data, which is saved as `model.pkl`. The Flask app allows users to input their loan details and receive a prediction about their loan status (approved or rejected).

## Table of Contents 📑

- [Project Overview](#project-overview-)
- [Features](#features-)
- [Requirements](#requirements-)
- [Installation](#installation-)
- [Usage](#usage-)
- [Project Structure](#project-structure-)
- [Contributing](#contributing-)

## Project Overview 📊

This Flask web app allows users to input loan application data, such as income, education, loan amount, etc., and receive a prediction of loan approval based on a pre-trained model.

The model is based on a **Random Forest Classifier** and trained on the following features:

- **Number of Dependents** 👨‍👩‍👧‍👦
- **Education Level** 🎓
- **Self-employed Status** 💼
- **Income (Annually)** 💰
- **Loan Amount** 💵
- **Loan Term** 📅
- **CIBIL Score** 🏦
- **Residential Assets Value** 🏠
- **Commercial Assets Value** 💼
- **Luxury Assets Value** 💎
- **Bank Asset Value** 💳

### Features 🌟

- **User Input via Web Form**: The Flask app allows users to input loan details through a form.
- **Prediction Output**: The app provides a **loan approval status** (Approved/Rejected) based on the entered data.
- **Model Integration**: The pre-trained **Random Forest Classifier** model predicts the loan status.

## Requirements 📦

To run this project, you need the following Python libraries:

- **Flask**: To create the web interface and serve the app.
- **pandas**: For data handling and preprocessing.
- **numpy**: For numerical operations.
- **scikit-learn**: To train the Random Forest Classifier model.
- **pickle**: For loading the saved model.

To install the required libraries, use:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install flask pandas numpy scikit-learn pickle-mixin
```

## Installation 🛠️

### Step 1: Clone the Repository

Clone this repository to your local machine:

```bash
git clone https://github.com/AdityaTagde/loan-approval-prediction.git
cd loan-approval-prediction
```

### Step 2: Install Dependencies

Install the required libraries:

```bash
pip install -r requirements.txt
```

### Step 3: Prepare the Dataset

Make sure you have the **`loan_approval_dataset.csv`** file in the root folder of the project (this is used for training the model).

### Step 4: Train the Model (If Not Already Done)

If you haven't trained the model yet, run the **training script** (`train_model.py`) to train the model and save it:

```bash
python train_model.py
```

This will train the model and save it as **`model.pkl`**, which is loaded by the Flask app.

### Step 5: Run the Flask App

To start the Flask web application, run the following command:

```bash
python app.py
```

This will start the server at `http://127.0.0.1:5000/`.

## Usage 📝

1. Open a web browser and navigate to `http://127.0.0.1:5000/`.
2. You will see a form where you can input the following loan details:
   - **Number of Dependents**
   - **Education Level**
   - **Self-employed Status**
   - **Income (Annually)**
   - **Loan Amount**
   - **Loan Term**
   - **CIBIL Score**
   - **Residential Assets Value**
   - **Commercial Assets Value**
   - **Luxury Assets Value**
   - **Bank Asset Value**
3. After filling in the form, click the **"Submit"** button.
4. The application will predict the loan status and display whether the loan is **Approved** or **Rejected**.

### Flask App:

- **`app.py`**: This is the main Flask application. It contains two routes:
  1. **Home Route (`/`)**: Displays the loan prediction form.
  2. **Prediction Route (`/predict`)**: Receives the form data, processes it, and returns the loan prediction.

## 🖼️ Screenshots
Input:
![App Screenshot](https://github.com/AdityaTagde/Loan_Approval_Prediction/blob/main/ip1.png)
![App Screenshot](https://github.com/AdityaTagde/Loan_Approval_Prediction/blob/main/ip2.png)
Output:
![App Screenshot](https://github.com/AdityaTagde/Loan_Approval_Prediction/blob/main/op1.png)
![App Screenshot](https://github.com/AdityaTagde/Loan_Approval_Prediction/blob/main/op2.png)

## Project Structure 📂

```
loan-approval-prediction/
│
├── app.py                  # Flask application code
├── train_model.py          # Model training script 🤖
├── loan_approval_dataset.csv  # Dataset 📊
├── model.pkl               # Pickled trained model 📦
├── requirements.txt        # List of required libraries 📜
└── README.md               # This file 📖
```

## Contributing 🤝

Feel free to fork the repository, create a new branch, and submit a pull request if you'd like to contribute! 😊
