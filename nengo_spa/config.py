from nengo.config import Config
from nengo.exceptions import ValidationError
from nengo.params import Parameter


class ConfigParam(Parameter):
    """A parameter where the value is a `.Config` object."""

    def validate(self, instance, value):
        super(ConfigParam, self).validate(instance, value)

        if value is not None and not isinstance(value, Config):
            raise ValidationError(
                "Must be of type 'Config' (got type %r)."
                % type(value).__name__, attr=self.name, obj=instance)

    def __set__(self, instance, value):
        if value is None:
            value = Config()
        super(ConfigParam, self).__set__(instance, value)
