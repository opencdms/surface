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
    # set utc_offset label
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

    # configure wigos section options
    wigos_part_1 = forms.IntegerField(initial='0', disabled=True, required=False, label='WIGOS ID Series')
    wigos_part_2 = forms.CharField(max_length=16, required=False, label='Issuer of identifier')
    wigos_part_3 = forms.IntegerField(min_value=0, max_value=65534, required=False, label='Issue Number')
    wigos_part_4 = forms.CharField(max_length=16, required=False, label='Local Identifier')

    class Meta:
        model = Station
        fields = '__all__'
        labels = {
            'wigos': 'WIGOS ID',
            'code': 'Station ID',
        }

    def clean(self):
        cleaned_data = super().clean()

        # Get the individual parts of the WIGOS ID
        wigos_part_1 = cleaned_data.get('wigos_part_1')
        wigos_part_2 = cleaned_data.get('wigos_part_2')
        wigos_part_3 = cleaned_data.get('wigos_part_3')
        wigos_part_4 = cleaned_data.get('wigos_part_4')

        # Combine the parts into a single string separated by '-'
        if wigos_part_1 and wigos_part_2 and wigos_part_3 and wigos_part_4:
            wigos_combined = f"{wigos_part_1}-{wigos_part_2}-{wigos_part_3}-{wigos_part_4}"

            # Set the combined value to the 'wigos' field
            cleaned_data['wigos'] = wigos_combined

        return cleaned_data


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