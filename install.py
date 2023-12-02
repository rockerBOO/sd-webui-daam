import launch


def check_matplotlib():
    if not launch.is_installed("matplotlib"):
        return False

    try:
        import matplotlib
    except ImportError:
        return False

    if hasattr(matplotlib, "__version_info__"):
        version = matplotlib.__version_info__
        version = (version.major, version.minor, version.micro)
        return version >= (3, 6, 2)
    return False


if not check_matplotlib():
    launch.run_pip(
        "install matplotlib==3.6.2", desc="Installing matplotlib==3.6.2"
    )


def check_daam():
    if not launch.is_installed("daam"):
        return False

    try:
        import daam
    except ImportError:
        return False

    if hasattr(daam, "_version"):
        version = daam.__version__.split(".")
        version = (int(version[0]), int(version[1]), int(version[2]))
        return version >= (0, 2, 0)
    return False


if not check_daam():
    launch.run_pip(
        "install git+https://github.com/rockerBOO/daam",
        desc="DAAM library",
    )
