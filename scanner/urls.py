# -*- coding: utf-8 -*-

from django.urls import path
from django.conf.urls import url
from . import views
urlpatterns = [
    url(r'sendvideofeed', views.get_videofeed, name='send_video'),

] 