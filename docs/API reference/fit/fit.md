## About fitting with `sqil_core`

In the `sqil_core` library, most fit functions use the [`@fit_output`](./core.md#sqil_core.fit._core.fit_output) decorator. This means that they return [`FitResult`](./core.md#sqil_core.fit._core.FitResult) object, even if the `return` statement in the source code suggests otherwise.

The `@fit_output` decorator automatically computes useful metrics like standard errors and chi squared values, while also providing tools to visualize the fit result.


You can also apply this decorator to your own custom fit functions. Simply import `@fit_output` from `sqil_core`, apply it to your function, and enjoy the same benefits.

For more information checkout the [core section](./core.md).

:::sqil_core.fit._fit
