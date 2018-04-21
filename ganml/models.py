from django.db import models

# Create your models here.
class Talk(models.Model):
  topic = models.TextField()
  speaker = models.TextField()
  scheduled_at = models.DateTimeField()

class TimeSeries(models.Model):
  year=models.DateTimeField()
  month=models.BigIntegerField()
  