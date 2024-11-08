### Loan Approval Prediction API
This project implements a machine learning model using XGBoost to predict loan eligibility based on clients' financial data. The model is served as a REST API using FastAPI, making it accessible in a production environment. The API evaluates prospective clients' eligibility for a loan based on their financial information.

## Project Structure
loan_approval.py: Contains model training, feature selection, and testing code.
evaluation.py: The main FastAPI application that serves the trained model as an API.
loan_model.pkl: The trained XGBoost model saved as a pickle file.
README.md: Documentation for the project.

## Requirements
Python 3.9
FastAPI
Uvicorn (for serving the FastAPI application)
Pandas
Scikit-learn
XGBoost
Pydantic

Install the dependencies:

```sh
pip install -r requirements.txt
```

## Model Training and Feature Selection
The dataset includes clients' financial data and loan approval status. Key features for prediction include:

loan_term: Loan term in months.
income_annum: Annual income.
cibil_score: Credit score.
residential_assets_value: Value of residential assets.
luxury_assets_value: Value of luxury assets.
loan_amount: Loan amount.

## Feature Selection Methods
To identify the most influential features, the following methods were applied in loan_approval.py:

1- Chi-Square Test: For evaluating categorical features.
2- ANOVA Test: To assess feature importance based on variance.
3- Correlation Heatmap: To detect multi-collinearity among features.
4- XGBoost Feature Importance (Gain): For feature ranking based on gain.
5- Mutual Information: To measure the dependency between features and the target variable.
The selected features were then used to train the XGBoost model, which was saved as 'loan_model.pkl' for use in the API.

##  Running the FastAPI Application:

### Using Docker:

#### 1. Clone the repository:

```sh
git clone <repository_url>
```

#### 2. Navigate to the App Directory:

```sh
cd <repository_name>
```

#### 3. Build the Docker image:

```sh
docker build -t <YOUR-IMAGE-NAME> .
```

#### 4. Run the container:
```sh
docker run -p 8081:8081 <YOUR-IMAGE-NAME>
```

## Usage
If not using Docker, you can still run the FastAPI application locally:

```sh
uvicorn evaluation:app --host 0.0.0.0 --port 8081
```
The API endpoint is available at http://localhost:8081/evaluation.

## API Endpoints
# POST /evaluation
This endpoint takes in client financial data and returns a loan eligibility prediction.

Request Body
loan_term (int)
income_annum (int)
cibil_score (float)
residential_assets_value (float)
luxury_assets_value (float)
loan_amount (float)
Example Request

```json
    {
    "loan_term": 24,
    "income_annum": 50000,
    "cibil_score": 750.0,
    "residential_assets_value": 300000.0,
    "luxury_assets_value": 100000.0,
    "loan_amount": 20000.0
    }
```
Example Response
- Status: 200 OK
- Body: "APPROVED" or "REJECTED"

## Testing
You can test the API with curl or Postman. Here’s an example using curl:

```sh
curl -X POST "http://0.0.0.0:8081/evaluation" -H "Content-Type: application/json" -d "{\"loan_term\":24, \"income_annum\":50000, \"cibil_score\":750.0, \"residential_assets_value\":300000.0, \"luxury_assets_value\":100000.0, \"loan_amount\":20000.0}"
```