import inspect


def debug_print(func, message="empty message", stop=True):
    if stop:
        input("\nDEBUG - {}.{}(): {}. Hit enter to continue\n".format("/".join((inspect.getfile(func).split("/"))[-2:]), func.__name__, message))
    else:
        print("\nDEBUG - {}.{}(): {}\n".format("/".join((inspect.getfile(func).split("/"))[-2:]), func.__name__, message))
