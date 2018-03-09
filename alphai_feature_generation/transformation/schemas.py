import datetime
from collections import defaultdict

from marshmallow import Schema, fields, validates, ValidationError


class AttributeDict(defaultdict):
    def __init__(self, *args, **kwargs):
        super(AttributeDict, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


# class AttributeDict(dict):
#     __getattr__ = dict.get
#     __setattr__ = dict.__setitem__
#     __delattr__ = dict.__delitem__


class BaseSchema(Schema):
    pass


class TimeDeltaSchema(BaseSchema):
    unit = fields.String()
    value = fields.Float()



class DataTransformationConfigurationSchema(BaseSchema):
    calendar_name = fields.String()
    features_ndays = fields.Integer()
    features_resample_minutes = fields.Integer()
    features_start_market_minute = fields.Integer()
    prediction_market_minute = fields.Integer()
    target_delta = fields.Raw()  # because it's already a timedelta object
    target_market_minute = fields.Integer()
    classify_per_series = fields.Boolean()
    normalise_per_series = fields.Boolean()
    n_classification_bins = fields.Integer()
    n_assets = fields.Integer()
    fill_limit = fields.Integer()
    predict_the_market_close = fields.Boolean(default=False, missing=False)
    feature_config_list = fields.List(fields.Dict())

    @validates('target_delta')
    def validate_target_delta(self, data):
        if not isinstance(data, datetime.timedelta):
            raise ValidationError('%s is not a valid timedelta')


class FinancialDataTransformationConfigurationSchema(DataTransformationConfigurationSchema):
    clean_nan_from_dict = fields.Boolean(default=False, missing=False)



class InvalidConfigurationException(Exception):
    pass
