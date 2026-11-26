# Student Placement Prediction

A machine learning web application that predicts student placement outcomes based on student input data.

## Features

- Student placement prediction
- Flask web application
- HTML and CSS frontend
- Trained machine learning model
- Joblib model loading
- Ready for deployment with Gunicorn

## Technologies Used

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- Joblib
- HTML
- CSS
- Gunicorn

## Project Structure

student-placement-prediction/
- app.py
- student_placement_best_model.joblib
- requirements.txt
- Procfile
- README.md
- .gitignore
- templates/
  - index.html
  - result.html
- static/
  - style.css

## How to Run

Install the required packages:

pip install -r requirements.txt

Run the application:

python app.py

The application will start on the Flask development server.

## Deployment

This project is prepared for deployment using Gunicorn and can be hosted on a cloud platform such as Render.

## Author

BCA Student
