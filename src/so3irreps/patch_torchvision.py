"""Patch torchvision import to work with torch versions that lack torchvision::nms.

Import this module BEFORE importing torchvision or lightning to avoid
``RuntimeError: operator torchvision::nms does not exist`` when the torch
build doesn't include the torchvision::nms operator.
"""

try:
    import torch._library.fake_impl as _fi
except ImportError:  # older torch (e.g. 2.4): no fake_impl module, nothing to patch
    _fi = None


def _patched_register(self, *args, **kwargs):
    try:
        return _orig_register(self, *args, **kwargs)
    except RuntimeError as e:
        if "does not exist" in str(e):
            # Return a no-op handle for operators that don't exist in this torch build.
            class _Dummy:
                def __enter__(self):
                    return None

                def __exit__(self, *a):
                    pass

                def destroy(self):
                    pass

            return _Dummy()
        raise


if _fi is not None:
    _orig_register = _fi.FakeImplHolder.register
    _fi.FakeImplHolder.register = _patched_register
