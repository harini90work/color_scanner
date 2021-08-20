# -*- coding: utf-8 -*-
from django.shortcuts import render
from django.http import HttpResponse

def home(request):
    '''
    Main Home function to display the web page

    Parameters
    ----------
    request : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        Render the Index.html.

    '''
    return render(request, 'index.html')

