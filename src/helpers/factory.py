import importlib

def create(cls, verbose=False):
    """
    Expects string that can be imported as with a module.class name
    """
    module_name, class_name = cls.rsplit(".", 1)

    try:
        if verbose: print(f"[*] importing {module_name}")
        somemodule = importlib.import_module(module_name)
        if verbose: print(f"[*] getattr {class_name}")
        cls_instance = getattr(somemodule, class_name)
        if verbose: print(cls_instance)

    except Exception as err:
        print(f"[-] Error creating factory: {err}")
        exit(-1)

    return cls_instance
