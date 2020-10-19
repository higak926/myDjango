import glob
import os
import pandas as pd

def get_file_list():
    file_list = os.listdir('./main/media/documents')
    return file_list

def get_order_list():
    order_list = ['未選択', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    return order_list

def get_item_list():
    item_list = ['customer_id', 'contract_type', 'sex', 'age', 'pref', 'payway', 'identified_doc',
                 'career', 'domain', 'region', 'entry_age', 'selling_method', 'campaign', 'amount',
                 'paid_flg', 'customer_rank', 'arrears_at', 'arrears_count', 'bill_updage_count']

    return item_list

def get_explanatory_variable_list():
    explanatory_variable_list = ['contract_type', 'age', 'payway', 'identified_doc', 'domain',
                                 'entry_age', 'amount', 'customer_rank', 'arrears_at', 'arrears_count',
                                 'bill_updage_count']

    return explanatory_variable_list

def get_join_file_list():
    join_file_list = [os.path.basename(p) for p in glob.glob('./static/output/*_join.csv')
                      if os.path.isfile(p)]

    return join_file_list

def get_model_result():
    model_result = [os.path.basename(p) for p in glob.glob('./static/output/model_result.csv')
                    if os.path.isfile(p)]

    return model_result

def get_score_file():
    score_file = [os.path.basename(p) for p in glob.glob('./static/output/score.csv')
                   if os.path.isfile(p)]

    return score_file

def get_score_list_display():
    score_list_display = []
    if os.path.exists('./static/output/score.csv'):
        scores_display = pd.read_csv("./static/output/score.csv").values.tolist()[0:10]
        for sl in scores_display:
            score_list_display.append(sl[1])

    return score_list_display

