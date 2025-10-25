def get_arg(args, kwargs, index, name, default=None):
    """
    Extracts an argument from either positional args or kwargs safely.

    Args:
        pos_args: tuple of positional arguments (*args)
        kwargs: dict of keyword arguments (**kwargs)
        index: positional index for this argument
        name: keyword name for this argument
        default: default value if neither present

    Returns:
        value: extracted value
    """
    # check positional args first
    if len(args) > index:
        # use positional argument
        value = args[index]
        # remove duplicate key from kwargs
        kwargs.pop(name, None)
    else:
        value = kwargs.pop(name, default)
    return value

def get_yaw(args, kwargs, index):
    return get_arg(args, kwargs, index, "yaw", default=0.0)

def get_tilt(args, kwargs, index):
    return get_arg(args, kwargs, index, "tilt", default=0.0)