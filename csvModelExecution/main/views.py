import csv
import io
import os
import pandas as pd
import boto3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from django.shortcuts import render
from .models import OtherInfo, Customer, Payment


def index(request):
    if request.method == 'POST':
        updata = request.FILES.getlist('file')

        for file in updata:
            fname = './main/media/documents/' + file.name

            f = open(fname, 'wb+')
            for chunk in file.chunks():
                f.write(chunk)
                f.close()

    file_list = os.listdir('./main/media/documents')
    return render(request, 'main/index.html', {'file_list': file_list})


def insert(request):
    file_list = os.listdir('./main/media/documents')

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

    return render(request, 'main/index.html', {'file_list': file_list})


def delete(request):
    file_list = os.listdir('./main/media/documents')
    Customer.objects.all().delete()
    Payment.objects.all().delete()
    OtherInfo.objects.all().delete()

    return render(request, 'main/index.html', {'file_list': file_list})

def join(request):
    file_list = os.listdir('./main/media/documents')
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

    if join == 'left-right-outer-join':
        outer_joined1 = pd.merge(customer_data, payment_data, how="outer", on="customer_id")
        outer_joined2 = pd.merge(outer_joined1, other_info_data, how="outer", on="customer_id")
        output_path = './static/output/'
        output_name = 'outer_join.csv'
        outer_joined2.to_csv(output_path + output_name)

    return render(request, 'main/index.html', {'file_list': file_list})

def model_create(request):
    file_list = os.listdir('./main/media/documents')
    df = pd.read_csv('./main/output/inner_join.csv')
    df_f = df.sample(frac=1)
    df_s = df_f.sample(n=5662)
    df_s.to_csv('./static/output/used_model_create.csv')
    df[~df.isin(df_s.to_dict(orient='list')).all(1)].to_csv('./main/output/data_25.csv')

    x1 = df_s[['customer_rank']]
    y1 = df_s['paid_flg']
    x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.2, random_state=0)
    lr = LogisticRegression()
    lr.fit(x1_train, y1_train)
    coefficient = lr.coef_
    intercept = lr.intercept_
    y1_pred = lr.predict(x1_test)

    confusion = confusion_matrix(y_true=y1_test, y_pred=y1_pred)
    accuracy = accuracy_score(y_true=y1_test, y_pred=y1_pred)
    precision = precision_score(y_true=y1_test, y_pred=y1_pred)
    recall = recall_score(y_true=y1_test, y_pred=y1_pred)
    f1 = f1_score(y_true=y1_test, y_pred=y1_pred)
    model_data_list = [[coefficient[0][0], intercept[0], [confusion[0][0], confusion[1][0]], accuracy, precision, recall, f1]]

    model = pd.DataFrame(model_data_list,
                         columns=['coefficient',
                                  'intercept',
                                  'confusion_matrix',
                                  'accuracy_score',
                                  'precision_score',
                                  'recall_score',
                                  'f1_score'
                                  ])
    model.to_csv('./static/output/model_result.csv')
    # TODO :  本番環境実行時に有効
    # s3 = boto3.resource('s3')
    # bucket = s3.Bucket('csv-model-execution')
    # bucket.upload_file('./static/output/model_result.csv')

    return render(request, 'main/index.html', {'file_list': file_list})
