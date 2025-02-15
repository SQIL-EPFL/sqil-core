import inspect


def _count_function_parameters(func):
    sig = inspect.signature(func)
    return len(
        [
            param
            for param in sig.parameters.values()
            if param.default == inspect.Parameter.empty
            and param.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.POSITIONAL_ONLY,
            )
        ]
    )
