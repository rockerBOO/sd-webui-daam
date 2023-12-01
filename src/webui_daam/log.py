def debug(*args, **kwargs):
    print("DAAM Debug: ", *args, **kwargs)


def info(message):
    print(f"DAAM: {message}")


def error(err, message):
    print(err)
    log(message)

    import traceback

    traceback.print_stack()


def warning(err, message):
    log(f"{err} {message}")


def log(*args, **kwargs):
    print(*args, **kwargs)
