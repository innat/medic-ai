import re


def hide_warnings():
    import logging
    import os
    import sys
    import warnings

    # Disable Python warnings
    def warn(*args, **kwargs):
        pass

    warnings.warn = warn
    warnings.simplefilter(action="ignore")
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Disable Python logging
    logging.disable(logging.WARNING)
    logging.getLogger("tensorflow").disabled = True

    # TensorFlow environment variables
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "3"

    # Disable TensorFlow debugging information
    if "tensorflow" in sys.modules:
        tf = sys.modules["tensorflow"]
        tf.get_logger().setLevel("ERROR")
        tf.autograph.set_verbosity(0)

    # Disable ABSL (Ten1orFlow dependency) logging
    try:
        import absl.logging

        absl.logging.set_verbosity(absl.logging.ERROR)
        # Redirect ABSL logs to null
        absl.logging.get_absl_handler().python_handler.stream = open(os.devnull, "w")
    except (ImportError, AttributeError):
        pass


def camel_to_snake(name: str) -> str:
    # Step 1: Put underscore between lower-uppercase or digit-uppercase
    s1 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    # Step 2: Handle acronym + word boundary (e.g., "CE" + "Loss")
    s2 = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s1)
    return s2.lower()
