"""
Email Spam Classification Web Application

This script creates a Streamlit-based web application for classifying email texts as spam or not spam. 
It integrates machine learning models, visualizations, and dataset exploration functionalities in an 
interactive and user-friendly interface.

This script is used to launch the app on Streamlit locally (for local testing).

Features:
---------
1. **Project Description**:
   - Explains the purpose, approach, and methodology of the spam classification project.

2. **Dataset Visualization**:
   - Displays visualizations from the dataset, including statistical and exploratory plots.

3. **Model Evaluation**:
   - Provides a performance comparison of various machine learning models used in the project.

4. **Model Visualization**:
   - Displays visual assets, such as confusion matrices or feature importance plots, for the models.

5. **Model Prediction**:
   - Allows users to input email text and get predictions on whether it is spam or not using the Logistic Regression model.

6. **Background Customization**:
   - Includes a custom background image with adjustable opacity for an enhanced user experience.

Technical Details:
------------------
- **Models and Vectorizer**:
  - Uses a pre-trained Logistic Regression model for classification.
  - Employs TF-IDF (Term Frequency-Inverse Document Frequency) for feature extraction.

- **Libraries Used**:
  - Streamlit: For creating the interactive web app.
  - scikit-learn: For machine learning model and text vectorization.
  - Pandas: For handling data in tabular format.
  - PIL (Pillow): For handling and displaying images.
  - OS: For directory traversal and file handling.

- **File Structure**:
  - `Models/Logistic Regression_Text_Best_Model.pkl`: Logistic Regression model file.
  - `Models/tfidf_vectorizer.pkl`: TF-IDF vectorizer file.
  - `Plots/Dataset/`: Contains dataset visualization images.
  - `Plots/Models/Text/`: Contains model evaluation images.

Usage:
------
1. Run the script using the Streamlit command:
   ```
   streamlit run Streamlit_app.py
   ```
2. Use the sidebar to navigate through various sections of the app.
3. Input email text in the "Model Prediction" section to classify it as spam or not.

Note:
-----
Ensure the necessary model and vectorizer files are present in the specified paths before running the app.
"""

# Importing Required Libraries
import streamlit as st
import pickle
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image

# Setting the page title
# This title will only be visible when running the app locally.
# In the deployed app, the title will be displayed as "Title - Streamlit," where "Title" is the one we provide.
# If we don't set the title, it will default to "Streamlit."
st.set_page_config(page_title="Spam Classification")

# Load the model and vectorizer from file
model_path = "Models/Logistic Regression_Text_Best_Model.pkl"
vectorizer_path = "Models/tfidf_vectorizer.pkl"

# Load the Logistic Regression model
with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

# Load the TF-IDF vectorizer
with open(vectorizer_path, "rb") as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)


# Define a function for project description
def project_description():
    st.subheader("Project Description")
    st.write(
        """
        This project focuses on email spam classification, leveraging advanced machine learning 
        techniques to distinguish between spam and non-spam emails with high accuracy. The system 
        is built using the Logistic Regression model, a robust and interpretable linear classifier 
        widely used for binary classification tasks.
        
        ### Key Features:
        - **Input**: Plain text of the email message.
        - **Output**: A prediction label indicating whether the email is "Spam" or "Not Spam."
        - **Preprocessing**: Text data is preprocessed and transformed using the TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer, ensuring effective feature extraction from text.
        - **Model**: Logistic Regression, optimized for high precision and recall on imbalanced datasets.
        - **Performance**: Achieves an accuracy of **94.32%**, along with strong precision and recall scores, minimizing false positives and false negatives.

        ### Logistic Regression Model
        Logistic Regression is a statistical model that predicts the probability of a binary outcome 
        (e.g., spam or not spam) based on input features. It uses a sigmoid activation function to 
        map predicted values to a range between 0 and 1, which can then be thresholded to classify the data.

        ### Why Logistic Regression?
        - **Efficiency**: Computationally lightweight and fast to train.
        - **Interpretability**: Coefficients provide insights into feature importance.
        - **Scalability**: Works well with large datasets when combined with efficient feature extraction like TF-IDF.

        This project demonstrates the integration of data preprocessing, model training, and deployment 
        into a seamless Streamlit-based web application.
        """
    )


# Define a function for dataset visualization
def dataset_visualization():
    dataset_dir = "Plots/Dataset"
    st.subheader("Dataset Visualizations")

    # Traverse the dataset directory and display all images
    for filename in os.listdir(dataset_dir):
        if filename.endswith((".png", ".jpg", ".jpeg", ".gif")):
            img_path = os.path.join(dataset_dir, filename)
            title = filename.replace("_", " ").split(".")[
                0
            ]  # Replace underscores with spaces and remove file extension
            st.subheader(title)
            st.image(img_path)


# Function for Model Evaluation
def model_evaluation():
    st.subheader("Model Performance Metrics")

    # Define the data for the table
    data = {
        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "SVC",
            "KNN",
            "Naive Bayes",
            "Gradient Boosting",
            "AdaBoost",
            "MLP",
        ],
        "Accuracy (%)": [94.32, 87.5, 93.18, 82.95, 60.23, 84.09, 94.32, 92.05, 92.05],
        "Precision": [0.871, 0.774, 0.844, 0.783, 0.444, 0.675, 0.871, 0.862, 0.8],
        "Recall": [0.964, 0.857, 0.964, 0.643, 1.0, 0.964, 0.964, 0.893, 1.0],
        "F1 Score": [0.915, 0.814, 0.9, 0.706, 0.615, 0.794, 0.915, 0.877, 0.889],
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Display the DataFrame in a table
    st.table(df)


# Define a function for model visualization
def model_visualization():
    model_dir = "Plots/Models/Text"
    st.subheader("Model Visualizations")

    # Traverse the model directory and display all images
    for filename in os.listdir(model_dir):
        if filename.endswith((".png", ".jpg", ".jpeg", ".gif")):
            img_path = os.path.join(model_dir, filename)
            title = filename.replace("_", " ").split(".")[
                0
            ]  # Replace underscores with spaces and remove file extension
            st.subheader(title)
            st.image(img_path)


# Define a function for model prediction
def model_prediction(email_text):
    # Transform the email text using the loaded TF-IDF vectorizer
    email_features = tfidf_vectorizer.transform([email_text]).toarray()

    # Make the prediction using the Logistic Regression model
    prediction = model.predict(email_features)
    result = "Spam" if prediction[0] == 1 else "Not Spam"

    st.subheader("Prediction Result")
    st.write(f"The email is predicted to be: **{result}**")


# Main Function
def main():

    # Streamlit app
    st.title("Email Spam Classification")

    # Dropdown for selecting functionality
    option = st.sidebar.radio(
        "Select an option:",
        [
            "Project Description",
            "Dataset Visualization",
            "Model Evaluation",
            "Model Visualization",
            "Model Prediction",
        ],
    )

    if option == "Project Description":
        project_description()

    elif option == "Dataset Visualization":
        dataset_visualization()

    elif option == "Model Evaluation":
        model_evaluation()

    elif option == "Model Visualization":
        model_visualization()

    elif option == "Model Prediction":
        st.subheader("Input Email Text")

        # User input field for email text
        email_text = st.text_area("Email Text")

        if st.button("Predict"):
            if email_text:
                model_prediction(email_text)
            else:
                st.warning("Please enter the email text.")


# Function to include background image and opacity
def display_background_image(url, opacity):
    """
    Displays a background image with a specified opacity on the web app using CSS.

    Args:
    - url (str): URL of the background image.
    - opacity (float): Opacity level of the background image.
    """
    # Set background image using HTML and CSS
    st.markdown(
        f"""
        <style>
            body {{
                background: url('{url}') no-repeat center center fixed;
                background-size: cover;
                opacity: {opacity};
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# Running the main function
if __name__ == "__main__":

    # Call function to display the background image with opacity
    display_background_image(
        "https://static.vecteezy.com/ti/gratis-vector/p2/16058824-concept-van-e-mail-en-computer-virussen-recensie-de-concepten-van-internet-veiligheid-spam-en-e-marketing-aan-scherm-spam-e-mail-pop-up-waarschuwingen-vector.jpg",
        0.8,
    )

    # Call main function to run the app
    main()
