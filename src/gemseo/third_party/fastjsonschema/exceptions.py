# -*- coding: utf-8 -*-
class JsonSchemaException(ValueError):
    """
    Exception raised by validation function. Contains ``message`` with
    information what is wrong.
    """

    def __init__(self, message):
        super(JsonSchemaException, self).__init__(message)
        self.message = message


class JsonSchemaDefinitionException(JsonSchemaException):
    """
    Exception raised by generator of validation function.
    """
