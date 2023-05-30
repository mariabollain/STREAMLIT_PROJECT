import streamlit as st
import pandas as pd
import pickle
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC

# Loading iris dataset
iris = datasets.load_iris()

# X-y split
X = iris["data"]
y = iris["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Train different models
lin_reg = LinearRegression()
log_reg = LogisticRegression()
svc_ = SVC()

lin_reg_fit = lin_reg.fit(X_train, y_train)
log_reg_fit = log_reg.fit(X_train, y_train)
svc_fit = svc_.fit(X_train, y_train)

# Creating pickle files (to save the training models)
with open("lin_reg.pkl", "wb") as li:  # wb: write mode
    pickle.dump(lin_reg_fit, li)

with open("log_reg.pkl", "wb") as lo:
    pickle.dump(log_reg_fit, lo)

with open("svc_.pkl", "wb") as sv:
    pickle.dump(svc_fit, sv)

# Read pickle files in main
with open("lin_reg.pkl", "rb") as li:  # rb: read mode
    linear_regression = pickle.load(li)

with open("log_reg.pkl", "rb") as lo:
    logistic_regression = pickle.load(lo)

with open("svc_.pkl", "rb") as sv:
    support_vector_classifier = pickle.load(sv)

# Function to classify the result into plants (0=setosa, 1=versicolor, 2=virginica)
def classify(num):
    if num == 0:
        return (st.success("Iris setosa"), st.image('setosa.jpg'))
    elif num == 1:
        return (st.success("Iris versicolor"), st.image('versicolor.jpg'))
    elif num == 2:
        return (st.success("Iris virginica"), st.image('virginica.jpg'))

# Defining main function
def main():

    # Title
    st.title("Modeling IRIS with Streamlit")

    # Sidebar title
    st.sidebar.header("User input parameters")

    # Function for the user to put parameters in sidebar
    def user_input_parameters():
        sepal_length = st.sidebar.slider("Sepal length", 4.3, 7.9, 6.0) # label, min, max, default
        sepal_width = st.sidebar.slider("Sepal width", 2.0, 4.4, 3.0) # label, min, max, default
        petal_length = st.sidebar.slider("Petal length", 1.0, 6.9, 4.0) # label, min, max, default
        petal_width = st.sidebar.slider("Petal width", 0.1, 2.5, 1.0) # label, min, max, default
        data = {"sepal_length":sepal_length,
                "sepal_width":sepal_width,
                "petal_length":petal_length,
                "petal_width":petal_width}
        features_df = pd.DataFrame(data, index=["user"])
        return features_df

    df = user_input_parameters()

    # The user will also choose the model in the sidebar
    option = {"Linear regression", "Logistic regression", "SVM classifier"}
    model = st.sidebar.selectbox("Which model do you want to use?", option)

    # Show user input in main page
    st.subheader("User input parameters")
    st.subheader(model)
    st.write(df)

    # Create button for running the model
    if st.button("RUN"):
        if model == "Linear regression":
            result = round(linear_regression.predict(df)[0], 0)
            classify(result)
        elif model == "Logistic regression":
            result = logistic_regression.predict(df)[0]
            classify(result)
        else:
            result = support_vector_classifier.predict(df)[0]
            classify(result)

if __name__ == '__main__':
    main()