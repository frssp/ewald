"""macOS integration: the app name in the menu bar and Dock, Finder's "Open With" files.

Qt-free so the platform checks are testable anywhere; everything here is a no-op off macOS.
"""
from __future__ import annotations

import sys


def strip_psn(argv):
    """Drop the ``-psn_0_12345`` process serial number older macOS passes to apps started from Finder."""
    return [a for a in argv if not a.startswith("-psn_")]


def set_app_name(name: str = "EWALD") -> bool:
    """Show ``name`` instead of "Python" in the menu bar's application menu when run from a terminal.

    Writes CFBundleName into the main bundle's info dictionary through the Objective-C runtime
    (ctypes, no pyobjc), which Cocoa reads when the application object is created, so call it
    before ``QApplication``. Inside the .app bundle the name comes from Info.plist instead.
    Returns True when the name was set.
    """
    if sys.platform != "darwin" or getattr(sys, "frozen", False):
        return False
    try:
        import ctypes
        import ctypes.util

        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        ctypes.cdll.LoadLibrary(ctypes.util.find_library("Foundation"))
        objc.objc_getClass.restype = objc.sel_registerName.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = objc.sel_registerName.argtypes = [ctypes.c_char_p]
        p = ctypes.c_void_p

        def send(restype, *argtypes):
            # one exact prototype per call: objc_msgSend is not variadic on arm64
            return ctypes.CFUNCTYPE(restype, p, p, *argtypes)(("objc_msgSend", objc))

        cls, sel = objc.objc_getClass, objc.sel_registerName
        info = send(p)(send(p)(cls(b"NSBundle"), sel(b"mainBundle")), sel(b"infoDictionary"))
        # setObject:forKey: on an immutable dictionary raises an Objective-C exception, which
        # would abort the process, so only write into a mutable one
        if not info or not send(ctypes.c_byte, p)(info, sel(b"isKindOfClass:"), cls(b"NSMutableDictionary")):
            return False
        string = send(p, ctypes.c_char_p)
        value = string(cls(b"NSString"), sel(b"stringWithUTF8String:"), name.encode())
        key = string(cls(b"NSString"), sel(b"stringWithUTF8String:"), b"CFBundleName")
        send(None, p, p)(info, sel(b"setObject:forKey:"), value, key)
        return True
    except Exception:
        return False
