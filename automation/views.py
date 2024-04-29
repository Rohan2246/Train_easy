from django.shortcuts import render, redirect
from .forms import DatasetUploadForm, PreprocessingForm, AlgorithmSelectionForm, MetricSelectionForm, TrainingForm
from .models import *
from django.http import JsonResponse
from .preprocessing import preprocess_data
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from django.views.decorators.csrf import csrf_exempt


@login_required
def upload_dataset(request):
    if request.session.get('current_dataset_id'):
        del request.session['current_dataset_id']
    if request.method == 'POST':
        if request.POST.get('dataset_id'):
            request.session['current_dataset_id'] = request.POST.get('dataset_id')
            return redirect('preprocess_selection')
        form = DatasetUploadForm(request.POST, request.FILES)
        if form.is_valid():
            dataset = form.save(commit=False)
            dataset.user = request.user
            dataset.save()
            messages.success(request, 'Dataset uploaded successfully')
            request.session['current_dataset_id'] = dataset.id
            return redirect('preprocess_selection')
    else:
        form = DatasetUploadForm()
    my_datasets= Dataset.objects.filter(user=request.user).all()
    return render(request, 'upload_dataset.html', {'form': form, 'my_datasets': my_datasets})

@login_required
def view_dataset(request):
    datasets = Dataset.objects.all()
    return render(request, 'view_dataset.html', {'datasets': datasets})

# TO BE TAKEN CARE later
def upload_dataset_api(request):
    if request.method == 'POST':
        form = DatasetUploadForm(request.POST, request.FILES)
        if form.is_valid():
            dataset = form.save(commit=False)
            dataset.user = request.user
            dataset.save()
            messages.success(request, 'Dataset uploaded successfully')
            return redirect('preprocess_selection')
    else:
        form = DatasetUploadForm()
    return render(request, 'components/upload_dataset.html', {'form': form})

def view_dataset_api(request):
    datasets = Dataset.objects.all()
    return render(request, 'components/view_dataset.html', {'datasets': datasets})


# Preprocessing view
def preprocess_selection(request):
    # clear old preprocess id
    if request.session.get('current_preprocess_id'):
        del request.session['current_preprocess_id']
    # load the selected dataset
    did = request.session.get('current_dataset_id')
    form = PreprocessingForm(initial={'dataset': did})
    if request.method == 'POST':
        form = PreprocessingForm(request.POST)
        if form.is_valid():
            preprocessing = form.save(commit=False)
            preprocessing.user = request.user
            preprocessing.save()
            request.session['current_preprocess_id'] = preprocessing.id
            messages.success(request, 'Preprocessing steps selected successfully')
            return redirect('algorithm_selection')
    ctx = {'form': form}
    return render(request, 'preprocess_selection.html', ctx)
            
            
def algorithm_selection(request):
    if request.session.get('current_algorithm_id'):
        del request.session['current_algorithm_id']
    did = request.session.get('current_dataset_id')
    pid = request.session.get('current_preprocess_id')
    form = AlgorithmSelectionForm(initial={'dataset': did, 'preprocessing': pid})
    if request.method == 'POST':
        form = AlgorithmSelectionForm(request.POST)
        if form.is_valid():
            algorithm = form.save(commit=False)
            algorithm.user = request.user
            algorithm.save()
            request.session['current_algorithm_id'] = algorithm.id
            messages.success(request, 'Algorithm selected successfully')
            return redirect('metric_selection')
    ctx = {'form': form}
    return render(request, 'algorithm_selection.html', ctx)
    
    
def metric_selection(request):
    if request.session.get('current_metric_id'):
        del request.session['current_metric_id']
    did = request.session.get('current_dataset_id')
    pid = request.session.get('current_preprocess_id')
    aid = request.session.get('current_algorithm_id')
    form = MetricSelectionForm(initial={'dataset': did, 'preprocessing': pid, 'algorithm': aid})
    if request.method == 'POST':
        form = MetricSelectionForm(request.POST)
        if form.is_valid():
            metric = form.save(commit=False)
            metric.user = request.user
            metric.save()
            request.session['current_metric_id'] = metric.id
            messages.success(request, 'Metrics selected successfully')
            return redirect('training')
    ctx = {'form': form}
    return render(request, 'metric_selection.html', ctx)

def training(request):
    if request.session.get('current_training_id'):
        del request.session['current_training_id']
    did = request.session.get('current_dataset_id')
    pid = request.session.get('current_preprocess_id')
    aid = request.session.get('current_algorithm_id')
    mid = request.session.get('current_metric_id')
    form = TrainingForm(initial={'dataset': did, 'preprocessing': pid, 'algorithm': aid, 'metric': mid})
    if request.method == 'POST':
        form = TrainingForm(request.POST)
        if form.is_valid():
            metrics = MetricSelection.objects.get(id=mid)
            algorithms = AlgorithmSelection.objects.get(id=aid)
            
            training = form.save(commit=False)
            training.user = request.user
            training.metric = metrics
            training.algo = algorithms
            
            training.save()
            request.session['current_training_id'] = training.id
            messages.success(request, 'Training started successfully')
            return redirect('finalize')
    ctx = {'form': form}
    return render(request, 'training.html', ctx)
    
        
        
def finalize_pipeline(request):
    did = request.session.get('current_dataset_id')
    pid = request.session.get('current_preprocess_id')
    aid = request.session.get('current_algorithm_id')
    mid = request.session.get('current_metric_id')
    tid = request.session.get('current_training_id')
    dataset = Dataset.objects.get(id=did)
    preprocesses = Preprocessing.objects.get(id=pid)
    algorithms = AlgorithmSelection.objects.get(id=aid)
    metrics = MetricSelection.objects.get(id=mid)
    training = Training.objects.get(id=tid)
    user = request.user
    ctx = {
        'dataset': dataset,
        'preprocesses': preprocesses,
        'algorithms': algorithms,
        'metrics': metrics,
        'training': training
    }
    return render(request, 'finalize.html', ctx)

@csrf_exempt
def execute_pipeline(request):
    did = request.session.get('current_dataset_id')
    pid = request.session.get('current_preprocess_id')
    aid = request.session.get('current_algorithm_id')
    mid = request.session.get('current_metric_id')
    tid = request.session.get('current_training_id')
    dataset = Dataset.objects.get(id=did)
    preprocesses = Preprocessing.objects.get(id=pid)
    algorithms = AlgorithmSelection.objects.get(id=aid)
    metrics = MetricSelection.objects.get(id=mid)
    training = Training.objects.get(id=tid)
    user = request.user
    
    return f'''
    <h1> Pipeline executed successfully </h1>
    <p> Dataset: {dataset.name} </p>
    '''

