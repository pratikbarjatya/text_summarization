from pywebio.input import *
from pywebio.output import *
from pywebio import start_server
from pywebio.exceptions import SessionClosedException
import pandas  as pd
import pickle
import warnings
import argparse

warnings.filterwarnings("ignore")

with open('./pickledFiles/random_forest_model.pkl', 'rb') as f:
    random_forest_model = pickle.load(f)

with open('./pickledFiles/columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)


def prediction(prediction_df):
    query_ = pd.get_dummies(pd.DataFrame(prediction_df, index=[0]), prefix=['Sector', 'job_sim'],
                            columns=['Sector', 'job_sim'])
    query = query_.reindex(columns=model_columns, fill_value=0)
    result = list(random_forest_model.predict(query))
    final_result = round(result[0], 3)

    return final_result


def main():
    put_markdown(
        '''
        # Food-in-Dine review summary Web App 
        '''
        , lstrip=True
    )

    model_inputs = input_group(
        "Enter the following information",
        [
            textarea("Please provide the review for food-in-dine restaurant ", name='review', type=TEXT)
        ]
    )

    prediction_df = pd.DataFrame(data=[model_inputs['review']])
                                 #columns=['Text_review'])

    summaryReview = prediction(prediction_df)
    put_markdown("### Your review summary: ", summaryReview)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=8080)
    args = parser.parse_args()

    start_server(main, port=args.port)
