from django import forms
from django.conf import settings
from django.contrib.admin.widgets import FilteredSelectMultiple

from wx.models import Station, FTPServer, WxGroupPermission, WxPermission


class FTPServerForm(forms.ModelForm):
    password = forms.CharField(widget=forms.TextInput(attrs={"type": "password"}))

    class Meta:
        model = FTPServer
        fields = '__all__'


class StationForm(forms.ModelForm):
    utc_offset_minutes = forms.IntegerField(initial=settings.TIMEZONE_OFFSET)

    class Meta:
        model = Station
        fields = '__all__'
        labels = {
            'utc_offset_minutes': 'UTC Offset (min)'
        }
        help_texts = {
            'is_active': '<span><span style="color:red;">**</span>Click \'is_active\' to display station on the map.</span>',
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
