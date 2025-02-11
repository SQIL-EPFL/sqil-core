## About fitting with `sqil_core`

In the `sqil_core` library, most fit functions use the [`@fit_output`](./core.md#sqil_core.fit._core.fit_output) decorator. This means that they return a [`FitResult`](./core.md#sqil_core.fit._core.FitResult) object, even if the `return` statement in the source code suggests otherwise.

The `@fit_output` decorator automatically computes useful metrics like standard errors and chi squared values, while also providing tools to visualize the fit result.


Fit functions can also use the [`@fit_input`](./core.md#sqil_core.fit._core.fit_input) decorator, which helps format input parameters (like bounds) in a more readable way, and allows to keep selected parameters fixed during the optimizaion.


You can also apply these decorators to your own custom fit functions. Simply import `@fit_output` or `@fit_input` from `sqil_core`, apply them to your function, and enjoy the same benefits.

For more information checkout the [core section](./core.md).

:::sqil_core.fit._fit
