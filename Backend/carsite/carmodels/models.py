from django.db import models
from django.contrib.auth.models import User


class imagedb(models.Model):
    image = models.ImageField(default='default.jpg', upload_to='profile_pics')

    def __str__(self):
        return f'image'

    def get_image_name(self):
        usb = imagedb.objects.all().last()
        new_image = usb.image
        #url = self.image.url
        return new_image

