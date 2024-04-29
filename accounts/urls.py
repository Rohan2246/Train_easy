from django.urls import path
from . import views

urlpatterns = [
    path("c/login", views.customer_login, name="clogin"),  
    path("c/register", views.customer_register, name="cregister"),
    path('logout', views.logout_view, name='logout'),
    path('profile/create', views.create_profile, name='create_profile'),
    path('dashboard/create', views.dashboard, name='dashboard'),
    
]