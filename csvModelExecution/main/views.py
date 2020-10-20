import collections
import csv
import io
import os
import pandas as pd
import matplotlib
# import japanize_matplotlib
# バックエンドを指定
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from django.http import HttpResponse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from django.shortcuts import render
from .models import OtherInfo, Customer, Payment
from . import views_list

def index(request):
    if request.method == 'POST':
        updata = request.FILES.getlist('file')

        for file in updata:
            fname = './main/media/documents/' + file.name

            f = open(fname, 'wb+')
            for chunk in file.chunks():
                f.write(chunk)
                f.close()
    return render(request, 'main/index.html', {'file_list': views_list.get_file_list(),
                                               'order_list': views_list.get_order_list(),
                                               'item_list': views_list.get_item_list(),
                                               'join_file_list': views_list.get_join_file_list(),
                                               'model_result': views_list.get_model_result(),
                                               'score_file': views_list.get_score_file(),
                                               'explanatory_variable_list': views_list.get_explanatory_variable_list(),
                                               'score_list_display': views_list.get_score_list_display(),
                                               'roc_auc': views_list.get_roc_auc_score()})

def insert(request):
    for file in views_list.get_file_list():
        if file == 'customer.csv':
            with open('./main/media/documents/customer.csv') as f:
                for row in csv.reader(f):
                    if row[0] != 'id':
                        customer, created = Customer.objects.get_or_create(id=int(row[0]),
                                                                           contract_type=int(row[1]),
                                                                           sex=row[2],
                                                                           age=int(row[3]),
                                                                           pref=row[4],
                                                                           payway=int(row[5]),
                                                                           identified_doc=int(row[6]),
                                                                           career=row[7],
                                                                           domain=int(row[8]),
                                                                           region=row[9],
                                                                           entry_age=int(row[10]),
                                                                           selling_method=row[11],
                                                                           campaign=row[12]
                                                                           )
                        customer.save()

        if file == 'payment.csv':
            with open('./main/media/documents/payment.csv') as f:
                for row in csv.reader(f):
                    if row[0] != 'id':
                        payment, created = Payment.objects.get_or_create(id=int(row[0]),
                                                                         amount=int(row[1]),
                                                                         paid_flg=int(row[2]),
                                                                         )
                        payment.save()

        if file == 'other_info.csv':
            with open('./main/media/documents/other_info.csv') as f:
                for row in csv.reader(f):
                    if row[0] != 'id':
                        otherInfo, created = OtherInfo.objects.get_or_create(id=int(row[0]),
                                                                             customer_rank=int(row[1]),
                                                                             arrears_at=int(row[2]),
                                                                             arrears_count=int(row[3]),
                                                                             bill_updage_count=int(row[4]),
                                                                             )
                        otherInfo.save()

    return render(request, 'main/index.html', {'file_list': views_list.get_file_list(),
                                               'order_list': views_list.get_order_list(),
                                               'item_list': views_list.get_item_list(),
                                               'join_file_list': views_list.get_join_file_list(),
                                               'model_result': views_list.get_model_result(),
                                               'score_file': views_list.get_score_file(),
                                               'explanatory_variable_list': views_list.get_explanatory_variable_list(),
                                               'score_list_display': views_list.get_score_list_display(),
                                               'roc_auc': views_list.get_roc_auc_score()})

def delete(request):
    Customer.objects.all().delete()
    Payment.objects.all().delete()
    OtherInfo.objects.all().delete()

    return render(request, 'main/index.html', {'file_list': views_list.get_file_list(),
                                               'order_list': views_list.get_order_list(),
                                               'item_list': views_list.get_item_list(),
                                               'join_file_list': views_list.get_join_file_list(),
                                               'model_result': views_list.get_model_result(),
                                               'score_file': views_list.get_score_file(),
                                               'explanatory_variable_list': views_list.get_explanatory_variable_list(),
                                               'score_list_display': views_list.get_score_list_display(),
                                               'roc_auc': views_list.get_roc_auc_score()})

def join(request):
    if  not views_list.get_file_list():
        return render(request, 'main/index.html', {'file_list': views_list.get_file_list(),
                                                   'order_list': views_list.get_order_list(),
                                                   'item_list': views_list.get_item_list(),
                                                   'join_file_list': views_list.get_join_file_list(),
                                                   'model_result': views_list.get_model_result(),
                                                   'score_file': views_list.get_score_file(),
                                                   'explanatory_variable_list': views_list.get_explanatory_variable_list(),
                                                   'score_list_display': views_list.get_score_list_display(),
                                                   'roc_auc': views_list.get_roc_auc_score()})

    join_order = [0]*19
    for item in views_list.get_item_list():
        order = request.POST[item]
        if order == '未選択':
            order_warning = 'すべての項目順序を入力して送信してください'
            return render(request, 'main/index.html', {'file_list': views_list.get_file_list(),
                                                       'order_list': views_list.get_order_list(),
                                                       'item_list': views_list.get_item_list(),
                                                       'join_file_list': views_list.get_join_file_list(),
                                                       'model_result': views_list.get_model_result(),
                                                       'score_file': views_list.get_score_file(),
                                                       'explanatory_variable_list': views_list.get_explanatory_variable_list(),
                                                       'order_warning': order_warning,
                                                       'score_list_display': views_list.get_score_list_display(),
                                                       'roc_auc': views_list.get_roc_auc_score()})

        join_order[int(order)-1] = item

    join = request.POST['join']

    for file in views_list.get_file_list():
        if file == 'customer.csv':
            with open('./main/media/documents/customer.csv') as f:
                customer_list = []
                for row in csv.reader(f):
                    if row[0] != 'id':
                        customer_list.append(row)

            customer_data = pd.DataFrame(customer_list,
                                         columns=['customer_id',
                                                  'contract_type',
                                                  'sex',
                                                  'age',
                                                  'pref',
                                                  'payway',
                                                  'identified_doc',
                                                  'career',
                                                  'domain',
                                                  'region',
                                                  'entry_age',
                                                  'selling_method',
                                                  'campaign'
                                                  ])

        if file == 'payment.csv':
            with open('./main/media/documents/payment.csv') as f:
                payment_list = []
                for row in csv.reader(f):
                    if row[0] != 'id':
                        payment_list.append(row)

            payment_data = pd.DataFrame(payment_list,
                                        columns=['customer_id',
                                                 'amount',
                                                 'paid_flg'
                                                 ])

        if file == 'other_info.csv':
            with open('./main/media/documents/other_info.csv') as f:
                other_info_list = []
                for row in csv.reader(f):
                    if row[0] != 'id':
                        other_info_list.append(row)

            other_info_data = pd.DataFrame(other_info_list,
                                           columns=['customer_id',
                                                    'customer_rank',
                                                    'arrears_at',
                                                    'arrears_count',
                                                    'bill_updage_count'
                                                    ])
    if join == 'inner-join':
        inner1 = pd.merge(customer_data, payment_data, how="inner", on="customer_id")
        inner2 = pd.merge(inner1, other_info_data, how="inner", on="customer_id")
        inner_joined = inner2.reindex(join_order, axis='columns')
        output_path = './static/output/'
        output_name = 'inner_join.csv'
        inner_joined.to_csv(output_path + output_name)

    if join == 'left-join':
        left1 = pd.merge(payment_data, customer_data, how="left", on="customer_id")
        left2 = pd.merge(left1, other_info_data, how="left", on="customer_id")
        left_joined = left2.reindex(join_order, axis='columns')
        output_path = './static/output/'
        output_name = 'left_join.csv'
        left_joined.to_csv(output_path + output_name)

    if join == 'outer-join':
        outer1 = pd.merge(customer_data, payment_data, how="outer", on="customer_id")
        outer2 = pd.merge(outer1, other_info_data, how="outer", on="customer_id")
        outer_joined = outer2.reindex(join_order, axis='columns')
        output_path = './static/output/'
        output_name = 'outer_join.csv'
        outer_joined.to_csv(output_path + output_name)

    return render(request, 'main/index.html', {'file_list': views_list.get_file_list(),
                                               'order_list': views_list.get_order_list(),
                                               'item_list': views_list.get_item_list(),
                                               'join_file_list': views_list.get_join_file_list(),
                                               'model_result': views_list.get_model_result(),
                                               'score_file': views_list.get_score_file(),
                                               'explanatory_variable_list': views_list.get_explanatory_variable_list(),
                                               'score_list_display': views_list.get_score_list_display(),
                                               'roc_auc': views_list.get_roc_auc_score()})


def model_create(request):
    if not views_list.get_file_list():
        return render(request, 'main/index.html', {'file_list': views_list.get_file_list(),
                                                   'order_list': views_list.get_order_list(),
                                                   'item_list': views_list.get_item_list(),
                                                   'join_file_list': views_list.get_join_file_list(),
                                                   'model_result': views_list.get_model_result(),
                                                   'score_file': views_list.get_score_file(),
                                                   'explanatory_variable_list': views_list.get_explanatory_variable_list(),
                                                   'score_list_display': views_list.get_score_list_display(),
                                                   'roc_auc': views_list.get_roc_auc_score()})

    df = pd.read_csv('./static/output/inner_join.csv')
    combined_format = request.POST['combined-format']
    if combined_format == 'left':
        df = pd.read_csv('./static/output/left_join.csv')

    choice_variable_list = []
    for var in views_list.get_explanatory_variable_list():
        request_var = request.POST.get(var)
        if request_var != None:
            choice_variable_list.append(request_var)

    if not choice_variable_list:
        choice_var_warning = '一つ以上の説明変数を選択してください'
        return render(request, 'main/index.html', {'file_list': views_list.get_file_list(),
                                                   'order_list': views_list.get_order_list(),
                                                   'item_list': views_list.get_item_list(),
                                                   'join_file_list': views_list.get_join_file_list(),
                                                   'model_result': views_list.get_model_result(),
                                                   'score_file': views_list.get_score_file(),
                                                   'explanatory_variable_list': views_list.get_explanatory_variable_list(),
                                                   'choice_var_warning': choice_var_warning,
                                                   'score_list_display': views_list.get_score_list_display(),
                                                   'roc_auc': views_list.get_roc_auc_score()})

    x1 = df[choice_variable_list]
    y1 = df[['paid_flg']]
    x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.25, random_state=0)

    x1_train_reault = pd.DataFrame(x1_train,columns=choice_variable_list)
    x1_test_reault = pd.DataFrame(x1_test,columns=choice_variable_list)
    y1_train_reault = pd.DataFrame(y1_train,columns=['paid_flg'])
    y1_test_reault = pd.DataFrame(y1_test,columns=['paid_flg'])
    x1_train_reault.to_csv('./static/output/data/x1_train.csv')
    x1_test_reault.to_csv('./static/output/data/x1_test.csv')
    y1_train_reault.to_csv('./static/output/data/y1_train.csv')
    y1_test_reault.to_csv('./static/output/data/y1_test.csv')

    lr = LogisticRegression()
    lr.fit(x1_train, y1_train)
    coefficient = lr.coef_
    intercept = lr.intercept_
    model_data_list = [[coefficient, intercept]]
    model_datas = pd.DataFrame(model_data_list, columns=['coefficient', 'intercept'])
    model_datas.to_csv('./static/output/model_data.csv')

    y1_pred = lr.predict(x1_test)
    prob_csv = lr.predict_proba(x1_test)[:, 1].tolist()
    prob = lr.predict_proba(x1_test)[:, 1].round(3).tolist()
    score_pd = pd.DataFrame(prob_csv, columns=['score'])
    score_pd.to_csv('./static/output/score.csv')

    confusion = confusion_matrix(y_true=y1_test, y_pred=y1_pred)
    accuracy = accuracy_score(y_true=y1_test, y_pred=y1_pred)
    precision = precision_score(y_true=y1_test, y_pred=y1_pred)
    roc_auc = roc_auc_score(y_true=y1_test, y_score=prob)
    recall = recall_score(y_true=y1_test, y_pred=y1_pred)
    f1 = f1_score(y_true=y1_test, y_pred=y1_pred)
    performance_evaluation_list = [[confusion,
                                    accuracy,
                                    precision,
                                    roc_auc,
                                    recall,
                                    f1]]

    performance_evaluations = pd.DataFrame(performance_evaluation_list,
                         columns=['confusion_matrix',
                                  'accuracy_score',
                                  'precision_score',
                                  'roc_auc',
                                  'recall_score',
                                  'f1_score'
                                  ])
    performance_evaluations.to_csv('./static/output/performance_evaluations.csv')

    return render(request, 'main/index.html', {'file_list': views_list.get_file_list(),
                                               'order_list': views_list.get_order_list(),
                                               'item_list': views_list.get_item_list(),
                                               'join_file_list': views_list.get_join_file_list(),
                                               'model_result': views_list.get_model_result(),
                                               'score_file': views_list.get_score_file(),
                                               'explanatory_variable_list': views_list.get_explanatory_variable_list(),
                                               'score_list_display': views_list.get_score_list_display(),
                                               'roc_auc': views_list.get_roc_auc_score()})


def get_svg(request):
    x1 = ['0']*20
    y1 = [0]*20
    if os.path.exists('./static/output/score.csv'):
        score_list = pd.read_csv("./static/output/score.csv").values.tolist()
        scores = []
        for sl in score_list:
            sl[1] = f"{sl[1]:.3f}"
            scores.append(sl[1])
            count = collections.Counter(scores).most_common()
            for i, c in enumerate(count):
                if i == 20:
                    break
                x1[i] = c[0]
                y1[i] = c[1]

    plt.bar(x1, y1, width=0.7, color='#00d5ff')
    # plt.title('モデル適用結果', fontweight='bold', loc='left')
    plt.xticks(rotation=90)
    plt.ylim([0, 1000])
    if os.path.exists('./static/output/score.csv'):
        plt.ylim([0, int(count[0][1]) + 100])
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.tick_params(bottom=False)

    buf = io.BytesIO()
    plt.savefig(buf, format='svg', bbox_inches='tight')
    s = buf.getvalue()
    buf.close()

    plt.cla()  # グラフをリセット
    response = HttpResponse(s, content_type='image/svg+xml')

    return response

