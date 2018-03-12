import datetime

from marshmallow import Schema, fields, validates, ValidationError


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
    n_forecasts = fields.Integer(default=1, missing=1)

    @validates('target_delta')
    def validate_target_delta(self, data):
        if not isinstance(data, datetime.timedelta):
            raise ValidationError('%s is not a valid timedelta')


class FinancialDataTransformationConfigurationSchema(DataTransformationConfigurationSchema):
    clean_nan_from_dict = fields.Boolean(default=False, missing=False)


class InvalidConfigurationException(Exception):
    pass
