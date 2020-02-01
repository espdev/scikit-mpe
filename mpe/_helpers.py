# -*- coding: utf-8 -*-

import functools
import inspect


def set_module(name: str):
    """Replace __module__ for decorated object

    Parameters
    ----------
    name : str
        Module name

    Returns
    -------
    obj : Any
        The object with replaced __module__
    """

    def decorator(obj):
        obj.__module__ = name
        return obj
    return decorator


def singledispatch(func_stub):
    """Single dispatch wrapper with default implementation that raises the exception for invalid signatures
    """

    @functools.wraps(func_stub)
    @functools.singledispatch
    def _dispatch(*args, **kwargs):
        sign_args = ', '.join(f'{type(arg).__name__}' for arg in args)
        sign_kwargs = ', '.join(f'{kwname}: {type(kwvalue).__name__}' for kwname, kwvalue in kwargs.items())
        allowed_signs = ''
        i = 0

        for arg_type, func in _dispatch.registry.items():
            if arg_type is object:
                continue
            i += 1
            sign = inspect.signature(func)
            allowed_signs += f'  [{i}] => {sign}\n'

        raise TypeError(
            f"call '{func_stub.__name__}' with invalid signature:\n"
            f"  => ({sign_args}, {sign_kwargs})\n\n"
            f"allowed signatures:\n{allowed_signs}")

    return _dispatch
