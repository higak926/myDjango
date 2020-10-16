import csv
import glob
import io
import os
import pandas as pd
import boto3
import matplotlib
# import japanize_matplotlib
# バックエンドを指定
from django.http import HttpResponse

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from django.shortcuts import render
from .models import OtherInfo, Customer, Payment

file_list = os.listdir('./main/media/documents')
explanatory_variable_list = ['contract_type',
                        'age',
                        'payway',
                        'identified_doc',
                        'domain',
                        'entry_age',
                        'amount',
                        'customer_rank',
                        'arrears_at',
                        'arrears_count',
                        'bill_updage_count']
join_file_list = [os.path.basename(p) for p in glob.glob('./static/output/*_join.csv')
                  if os.path.isfile(p)]
model_result = [os.path.basename(p) for p in glob.glob('./static/output/model_result.csv')
                if os.path.isfile(p)]
model_apply = [os.path.basename(p) for p in glob.glob('./static/output/model_apply.csv')
               if os.path.isfile(p)]

def index(request):
    if request.method == 'POST':
        updata = request.FILES.getlist('file')

        for file in updata:
            fname = './main/media/documents/' + file.name

            f = open(fname, 'wb+')
            for chunk in file.chunks():
                f.write(chunk)
                f.close()
    return render(request, 'main/index.html', {'file_list': file_list,
                                               'join_file_list': join_file_list,
                                               'model_result': model_result,
                                               'model_apply': model_apply,
                                               'explanatory_variable_list': explanatory_variable_list})

def insert(request):
    for file in file_list:
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

    return render(request, 'main/index.html', {'file_list': file_list,
                                               'join_file_list': join_file_list,
                                               'model_result': model_result,
                                               'model_apply': model_apply,
                                               'explanatory_variable_list': explanatory_variable_list})

def delete(request):
    Customer.objects.all().delete()
    Payment.objects.all().delete()
    OtherInfo.objects.all().delete()

    return render(request, 'main/index.html', {'file_list': file_list,
                                               'join_file_list': join_file_list,
                                               'model_result': model_result,
                                               'model_apply': model_apply,
                                               'explanatory_variable_list': explanatory_variable_list})

def join(request):
    join = request.POST['join']

    for file in file_list:
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
        inner_joined1 = pd.merge(customer_data, payment_data, how="inner", on="customer_id")
        inner_joined2 = pd.merge(inner_joined1, other_info_data, how="inner", on="customer_id")
        output_path = './static/output/'
        output_name = 'inner_join.csv'
        inner_joined2.to_csv(output_path + output_name)

    if join == 'left-join':
        left_joined1 = pd.merge(payment_data, customer_data, how="left", on="customer_id")
        left_joined2 = pd.merge(left_joined1, other_info_data, how="left", on="customer_id")
        output_path = './static/output/'
        output_name = 'left_join.csv'
        left_joined2.to_csv(output_path + output_name)

    if join == 'outer-join':
        outer_joined1 = pd.merge(customer_data, payment_data, how="outer", on="customer_id")
        outer_joined2 = pd.merge(outer_joined1, other_info_data, how="outer", on="customer_id")
        output_path = './static/output/'
        output_name = 'outer_join.csv'
        outer_joined2.to_csv(output_path + output_name)

    return render(request, 'main/index.html', {'file_list': file_list,
                                               'join_file_list': join_file_list,
                                               'model_result': model_result,
                                               'model_apply': model_apply,
                                               'explanatory_variable_list': explanatory_variable_list})

def model_create(request):
    df = pd.read_csv('./static/output/inner_join.csv')
    combined_format = request.POST['combined-format']
    if combined_format == 'left':
        df = pd.read_csv('./static/output/left_join.csv')

    # df_f = df.sample(frac=1)
    # df_s = df_f.sample(n=5662)
    # df_s.to_csv('./main/output/used_model_create.csv')
    # df[~df.isin(df_s.to_dict(orient='list')).all(1)].to_csv('./main/output/data_25.csv')
    explanatory_var = request.POST['model-create']
    combined_format = request.POST['combined-format']
    breakpoint()

    x1 = df[[explanatory_var]]
    y1 = df[['paid_flg']]
    x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.25, random_state=0)
    # TODO : model_list各々要作成
    model_list = [[x1_train, x1_test, y1_train, y1_test]]
    model = pd.DataFrame(model_list,
                          columns=['x1_train',
                                   'x1_test',
                                   'y1_train',
                                   'y1_test'])
    model.to_csv('./static/output/model.csv')

    lr = LogisticRegression()
    lr.fit(x1_train, y1_train)
    coefficient = lr.coef_
    intercept = lr.intercept_
    y1_pred = lr.predict(x1_test)
    prob = lr.predict_proba(x1_test)[:, 1]
    prob_list = []
    for i, el in enumerate(prob):
        prob_list += [[i, el]]
    model_apply_result = pd.DataFrame(prob_list, columns=['id', 'score'])
    model_apply_result.to_csv('./static/output/model_apply_result.csv')

    confusion = confusion_matrix(y_true=y1_test, y_pred=y1_pred)
    accuracy = accuracy_score(y_true=y1_test, y_pred=y1_pred)
    precision = precision_score(y_true=y1_test, y_pred=y1_pred)
    roc_auc = roc_auc_score(y_true=y1_test, y_score=prob)
    recall = recall_score(y_true=y1_test, y_pred=y1_pred)
    f1 = f1_score(y_true=y1_test, y_pred=y1_pred)
    performance_evaluation_list = [[coefficient[0][0],
                                    intercept[0],
                                    [[confusion[0][0], confusion[0][1]], [confusion[1][0], confusion[1][1]]],
                                    accuracy,
                                    precision,
                                    roc_auc,
                                    recall,
                                    f1]]

    performance_evaluations = pd.DataFrame(performance_evaluation_list,
                         columns=['coefficient',
                                  'intercept',
                                  'confusion_matrix',
                                  'accuracy_score',
                                  'precision_score',
                                  'roc_auc',
                                  'recall_score',
                                  'f1_score'
                                  ])
    performance_evaluations.to_csv('./static/output/performance_evaluations.csv')

    return render(request, 'main/index.html', {'file_list': file_list,
                                               'join_file_list': join_file_list,
                                               'model_result': model_result,
                                               'model_apply': model_apply,
                                               'explanatory_variable_list': explanatory_variable_list})


def get_svg(request):
    x1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    x = ['0.047', '0.094', '0.140', '0.187', '0.234', '0.281', '0.328', '0.374', '0.421', '0.468', '0.515', '0.562',
         '0.608', '0.655', '0.702', '0.749', '0.795', '0.842', '0.889', '0.936', '0.983']
    y = [16150, 2270, 1088, 526, 245, 195, 114, 65, 68, 43, 34, 30, 32, 29, 19, 16, 23, 23, 16, 13, 1]
    plt.bar(x, y, width=0.7, color='#00d5ff')
    # plt.title('モデル適用結果', fontweight='bold', loc='left')
    plt.xticks(rotation=90)
    plt.ylim([0, 18000])
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

def model_apply(request):
    model_list = []
    model = pd.DataFrame(model_list,
                         columns=['coefficient'])
    model.to_csv('./static/output/model_apply.csv')

    return render(request, 'main/index.html', {'file_list': file_list,
                                               'join_file_list': join_file_list,
                                               'model_result': model_result,
                                               'model_apply': model_apply,
                                               'explanatory_variable_list': explanatory_variable_list})

def csv_download(request):
    model_list = []
    model = pd.DataFrame(model_list,
                         columns=['coefficient'])
    response = HttpResponse(model, content_type='text/csv')
    filename = 'model_apply.csv'
    response['Content-Disposition'] = 'attachment; filename={}'.format(filename)

    return response
