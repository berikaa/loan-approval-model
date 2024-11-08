import pandas as pd
from fastapi import FastAPI, HTTPException
import pickle
import uvicorn
from pydantic import BaseModel, conint, Field, confloat

app = FastAPI()
try:
    with open("loan_model.pkl", "rb") as model_file:
        xgb_model = pickle.load(model_file)
except FileNotFoundError:
    raise Exception("Model file 'loan_model.pkl' not found.")
except pickle.UnpicklingError:
    raise Exception("An error occurred while loading the model file 'loan_model.pkl'.")


class EvaluationInput(BaseModel):
    loan_term: conint(ge=0) = Field(..., description="Loan term must be a non-negative integer.")
    income_annum: conint(ge=0) = Field(..., description="Annual income must be a non-negative integer.")
    cibil_score: confloat(ge=300, le=900) = Field(
        ..., description="CIBIL score must be between 300 and 900."
    )
    residential_assets_value: confloat(ge=0) = Field(
        ..., description="Residential assets value must be a non-negative number."
    )
    luxury_assets_value: confloat(ge=0) = Field(
        ..., description="Luxury assets value must be a non-negative number."
    )
    loan_amount: confloat(ge=0) = Field(
        ..., description="Loan amount must be a non-negative number."
    )
@app.post("/evaluation")
def evaluation(input_data: EvaluationInput):
    try:
        data = {
            'loan_term': [input_data.loan_term],
            'income_annum': [input_data.income_annum],
            'cibil_score': [input_data.cibil_score],
            'luxury_assets_value': [input_data.luxury_assets_value],
            'loan_amount': [input_data.loan_amount],
            'residential_assets_value': [input_data.residential_assets_value],
        }
        input_df = pd.DataFrame(data)

        try:
            prediction = xgb_model.predict(input_df)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An error occurred during model prediction: {e}")
               
        return {"status": "APPROVED" if prediction[0] == 1 else "REJECTED"}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid parameter value: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
