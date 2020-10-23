import glob
import os
import pandas as pd


def get_file_list():
    file_list = os.listdir('./main/media/documents')
    return file_list

def file_check():
    check = True
    files_list = ['customer.csv', 'payment.csv', 'other_info.csv']
    files = set(files_list)
    if files == set(get_file_list()):
        check = False
    return check

def get_explanatory_variable_list():
    explanatory_variable_list = ['customer_id', 'contract_type', 'sex', 'age', 'pref', 'payway', 'identified_doc',
                 'career', 'domain', 'region', 'entry_age', 'selling_method', 'campaign', 'amount',
                 'customer_rank', 'arrears_at', 'arrears_count', 'bill_updage_count']
    return explanatory_variable_list

def get_join_file_list():
    join_file_list = [os.path.basename(p) for p in glob.glob('./static/output/*_join.csv')
                      if os.path.isfile(p)]
    return join_file_list

def get_model_result():
    model_result = [os.path.basename(p) for p in glob.glob('./static/output/model_data.csv')
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

def get_roc_auc_score():
    roc_auc_score = ''
    if os.path.exists('./static/output/performance_evaluations.csv'):
        roc_auc_score = pd.read_csv("./static/output/performance_evaluations.csv", usecols=['auc']).values[0][0]
    return roc_auc_score
