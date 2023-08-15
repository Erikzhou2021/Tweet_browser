#!/usr/bin/env python
# coding: utf-8

# Copyright (c) Erik Zhou.
# Distributed under the terms of the Modified BSD License.

import pytest

from ..widget import ExampleWidget


def test_example_creation_blank():
    w = ExampleWidget()
    assert w.value == 'Hello World'
