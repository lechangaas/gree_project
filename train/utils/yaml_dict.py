"""
[INTEL CONFIDENTIAL]

Copyright (c) 2019 Intel Corporation.

This software and the related documents are Intel copyrighted materials, and
your use of them is governed by the express license under which they were 
provided to you ("License"). Unless the License provides otherwise, you may
not use, modify, copy, publish, distribute, disclose or transmit this
software or the related documents without Intel's prior written permission.

This software and the related documents are provided as is, with no express
or implied warranties, other than those that are expressly stated in the License.
"""

class YAMLDict(dict):
    def __init__(self, dt=None, **kwargs):
        if dt is None:
            dt = {}
        if kwargs:
            dt.update(**kwargs)
        for key, value in dt.items():
            setattr(self, key, value)
        for key in self.__class__.__dict__.keys():
            if not (key.startswith('__') and key.endswith('__')) and not key  in ('update'):
                setattr(self, key, getattr(self, key))

    def __setattr__(self, n, v):
        if isinstance(v, (list, tuple)):
            v = [self.__class__(i)
                     if isinstance(i, dict) else i for i in v]
        elif isinstance(v, dict) and not isinstance(v, self.__class__):
            v = self.__class__(v)
        super(YAMLDict, self).__setattr__(n, v)
        super(YAMLDict, self).__setitem__(n, v)

    __setitem__ = __setattr__

    def update(self, y=None, **f):
        dt = y or dict()
        dt.update(f)
        for key in dt:
            setattr(self, key, dt[key])
