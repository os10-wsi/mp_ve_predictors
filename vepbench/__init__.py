"""vepbench: saturation variant-effect prediction across many predictors."""

__version__ = "0.1.0"

# Several public data sources reject the default user agent of common HTTP
# clients. AlphaFold DB in particular answers 403 to a bare curl or requests
# UA, which looks like a network block but is not one.
USER_AGENT = (
    "vepbench/0.1.0 (https://github.com/os10-wsi/mp_ve_predictors; "
    "membrane-protein variant-effect benchmarking)"
)
HTTP_HEADERS = {"User-Agent": USER_AGENT}
