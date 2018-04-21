from rest_framework import serializers
from .models import Talk,TimeSeries
class TalkSerializer (serializers.ModelSerializer):
  class Meta:
    model = Talk
    fields = ('topic', 'speaker', 'scheduled_at')


class TimeSeriesSerializer(serializers.ModelSerializer):
  class Meta:
    model=TimeSeries
    fields=('year','month')