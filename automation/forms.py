from django import forms
from .models import Dataset, Preprocessing, AlgorithmSelection, MetricSelection, Training

class DatasetUploadForm(forms.ModelForm):
    class Meta:
        model = Dataset
        fields = ['name', 'file']
        widgets = {
            'name': forms.TextInput(attrs={'placeholder': 'Enter Project Name'}),
        }
        
class PreprocessingForm(forms.ModelForm):
    class Meta:
        model = Preprocessing
        fields = ['dataset','normalization', 'encoding', 'imputation', 'feature_selection', 'pca']
        
class AlgorithmSelectionForm(forms.ModelForm):
    class Meta:
        model = AlgorithmSelection
        fields = ['dataset','linear_Regression', 'logistic_Regression', 'decision_Tree', 'random_Forest', 'support_Vector_Machines', 'naive_Bayes']
        
class MetricSelectionForm(forms.ModelForm): 
    class Meta:
        model = MetricSelection
        fields = ['dataset','accuracy', 'mse', 'rmse', 'mae', 'confusion_matrix', 'roc_auc', 'precision', 'recall', 'f1']
        
class TrainingForm(forms.ModelForm):
    class Meta:
        model = Training
        fields = ['dataset' ,'split' ]  
        
