#!/usr/bin/env python
# coding: utf-8

# Copyright (c) Erik Zhou.
# Distributed under the terms of the Modified BSD License.

from ipywidgets import DOMWidget, ValueWidget, register
from traitlets import Unicode, Bool, validate, TraitError

from ._frontend import module_name, module_version


@register
class Email(DOMWidget, ValueWidget):
    _model_name = Unicode('EmailModel').tag(sync=True)
    _model_module = Unicode(module_name).tag(sync=True)
    _model_module_version = Unicode(module_version).tag(sync=True)

    _view_name = Unicode('EmailView').tag(sync=True)
    _view_module = Unicode(module_name).tag(sync=True)
    _view_module_version = Unicode(module_version).tag(sync=True)

    value = Unicode('example@example.com').tag(sync=True)
    disabled = Bool(False, help="Enable or disable user changes.").tag(sync=True)

    # Basic validator for the email value
    @validate('value')
    def _valid_value(self, proposal):
        if proposal['value'].count("@") != 1:
            raise TraitError('Invalid email value: it must contain an "@" character')
        if proposal['value'].count(".") == 0:
            raise TraitError('Invalid email value: it must contain at least one "." character')
        return proposal['value']
