import csv
import os

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
                                                                         paid_flg=int(row[2])
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
                                                                             bill_updage_count=int(row[4])
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
    Customer.objects.all().delete()
    Payment.objects.all().delete()
    OtherInfo.objects.all().delete()

    return render(request, 'main/index.html', {'file_list': file_list})


# messages.success(request, 'アップロードが完了しました。')

# if request.method == 'POST':
#   form = DocumentForm(request.POST, request.FILES)
#   if form.is_valid():
#      form.save()
#      messages.success(request, 'アップロードを保存しました。')
# else:
#   form = DocumentForm()
#   messages.warning(request, 'アップロードできませんでした')

# if request.method == 'POST':
    #     form = UploadFileForm(request.POST, request.FILES)
    #     if form.is_valid():
    #         handle_uploaded_file(request.FILES['file'])
    # else:
    #     form = UploadFileForm()
    # return render(request, 'main/index.html', {'form': form})

# def handle_uploaded_file(file_obj):
#     file_path = 'media/documents/' + file_obj.name
#     with open(file_path, 'wb+') as destination:
#         for chunk in file_obj.chunks():
#             destination.write(chunk)

    # content = {
    # 'message': 'CSVによるモデル実行システム'
    # }
    # return render(request, 'main/index.html', content)
