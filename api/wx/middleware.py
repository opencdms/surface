from asgiref.sync import iscoroutinefunction
from django.utils.decorators import sync_and_async_middleware
from .models import WxGroupPermission

@sync_and_async_middleware
def user_permissions_middleware(get_response):
    if iscoroutinefunction(get_response):
        async def middleware(request):
            if request.user.is_authenticated:
                user_groups = request.user.groups.all()
                permissions = await WxGroupPermission.objects.filter(group__in=user_groups).prefetch_related('permissions').aasync()

                user_permissions = set()
                for group_permission in permissions:
                    user_permissions.update(group_permission.permissions.values_list('url_name', flat=True))

                request.user_permissions = user_permissions
            else:
                request.user_permissions = set()

            response = await get_response(request)
            return response
    else:
        def middleware(request):
            if request.user.is_authenticated:
                user_groups = request.user.groups.all()
                permissions = WxGroupPermission.objects.filter(group__in=user_groups).prefetch_related('permissions')

                user_permissions = set()
                for group_permission in permissions:
                    user_permissions.update(group_permission.permissions.values_list('url_name', flat=True))

                request.user_permissions = user_permissions
            else:
                request.user_permissions = set()

            response = get_response(request)
            return response

    return middleware

            