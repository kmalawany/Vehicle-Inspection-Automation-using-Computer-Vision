from django.contrib import admin
from .models import profile, createprofile, Customer, damageinfo, report

admin.site.register(profile)
admin.site.register(createprofile)
admin.site.register(Customer)
admin.site.register(damageinfo)

admin.site.register(report)
