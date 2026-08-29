from django.contrib import admin
from django.urls import path, include
from . import views
from .embed_view import embed_chat_view
path('embed/', embed_chat_view, name='embed_chat'),
urlpatterns = [
    path('', views.home, name='home'),
    path('api/query/', views.query_view, name='query_view'),
    path('embed/', embed_chat_view, name='embed_chat'),
]
