"""TLS context for downloads (plugin indexes, example datasets).

Behind a company proxy that re-signs HTTPS, Python's own CA list fails with "certificate verify
failed" while the operating system trusts the proxy's root. ``truststore`` asks the OS store
(macOS, Windows, Linux); without it the default context (``SSL_CERT_FILE`` is honoured) is used.
"""
from __future__ import annotations

import ssl


def ssl_context() -> ssl.SSLContext:
    try:
        import truststore
    except ImportError:
        return ssl.create_default_context()
    return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
