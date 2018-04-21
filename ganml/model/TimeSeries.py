from rest_pandas import PandasSimpleView
from rest_pandas import  serializers

class TimeSeriesView(PandasView):
  queryset = TimeSeries.objects.all()
  serializer_class = TimeSeriesSerializer

class TimeSeriesView(PandasView):
    # Assign a default model queryset to the view
    queryset = TimeSeries.objects.all()

    # Step 1. In response to get(), the underlying Django REST Framework view
    # will load the queryset and then pass it to the following function.
    def filter_queryset(self, qs):
      # At this point, you can filter queryset based on self.request or other
      # settings (useful for limiting memory usage).  This function can be
      # omitted if you are using a filter backend or do not need filtering.
      return qs

    serializer_class = TimeSeriesSerializer
    pandas_serializer_class = MyCustomPandasSerializer

    def transform_dataframe(self, dataframe):
      dataframe.some_pivot_function(in_place=True)
      return dataframe


