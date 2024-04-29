from django.urls import path
from . import views

urlpatterns = [
    path('upload/dataset', views.upload_dataset, name='upload_dataset'),
    path('view/dataset', views.view_dataset, name='view_dataset'),
    path('preprocess/selection', views.preprocess_selection, name='preprocess_selection'),
    path('algorithm/selection', views.algorithm_selection, name='algorithm_selection'),
    path('metric/selection', views.metric_selection, name='metric_selection'),
    path('training', views.training, name='training'),
    path('finalize', views.finalize_pipeline, name='finalize'),
]

api_urlpatterns = [
    path('api/upload/dataset', views.upload_dataset_api, name='upload_dataset_api'),
    path('api/view/dataset', views.view_dataset_api, name='view_dataset_api'),
]


urlpatterns += api_urlpatterns