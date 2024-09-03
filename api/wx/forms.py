from django import forms
from django.conf import settings
from django.contrib.admin.widgets import FilteredSelectMultiple

from wx.models import Station, Watershed, AdministrativeRegion, FTPServer, WxGroupPermission, WxPermission

class FTPServerForm(forms.ModelForm):
    password = forms.CharField(widget=forms.TextInput(attrs={"type": "password"}))

    class Meta:
        model = FTPServer
        fields = '__all__'


class StationForm(forms.ModelForm):
    utc_offset_minutes = forms.IntegerField(
        label='UTC Offset (min)', 
        initial=settings.TIMEZONE_OFFSET
    )

    # configured dropdown for watershed option
    watershed_options = [('','---------')] # placeholder value
    for x in Watershed.objects.values_list('watershed', flat=True):
        watershed_options.append((x,x))

    watershed = forms.ChoiceField(choices=watershed_options)

    # configured dropdown for region option
    region_options = [('','---------')] # placeholder value
    for x in AdministrativeRegion.objects.values_list('name', flat=True):
        region_options.append((x,x))

    region = forms.ChoiceField(choices=region_options)

    class Meta:
        model = Station
        fields = '__all__'
        labels = {
            'wigos': 'WIGOS ID',
            'code': 'Station ID',
        }


class WxGroupPermissionForm(forms.ModelForm):
    permissions = forms.ModelMultipleChoiceField(
        queryset=WxPermission.objects.all(),
        required=True,
        widget=FilteredSelectMultiple(
            verbose_name='Permissions',
            is_stacked=False
        )
    )

    class Meta:
        model = WxGroupPermission
        fields = '__all__'