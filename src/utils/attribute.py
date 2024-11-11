
def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])
def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)
def get_attr(obj, names):
    if len(names) == 1:
        return getattr(obj, names[0])
    return get_attr(getattr(obj, names[0]), names[1:])