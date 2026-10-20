# Settings pages

A plugin that needs setting up (a server address, a login, an access key) can give itself a
settings page, so users do it in the GUI instead of in Python or a terminal. ewald builds the
dialog from a list of fields; the plugin only reads and stores the values. The page opens from
the plugin's **Settings** button in the Plugins side bar and from Edit › Plugin settings.

```python
from ewald.keychain import delete_secret, get_secret, set_secret
from ewald.plugin_api import SettingsField, settings_page


@settings_page("myservice", title="My service", fields=[
    SettingsField("url", "Server address", placeholder="https://server/"),
    SettingsField("key", "Access key", kind="secret", help="Made in the service's web UI"),
    SettingsField("verify", "Check certificates", kind="bool", default=True)],
    actions={"Log out": "logout"})
class MyServiceSettings:
    """Connect ewald to My service."""

    def load(self):
        cfg = read_config()                       # your own file, without secrets
        return {"url": cfg.get("url", ""), "verify": cfg.get("verify", True),
                "key": get_secret("myservice")[0] is not None}

    def save(self, values):
        if not values["url"].startswith("https://"):
            raise ValueError("The server address starts with https://")
        if values["key"]:                         # empty: keep the saved key
            set_secret("myservice", values["key"])
        write_config({"url": values["url"], "verify": values["verify"]})
        return "Saved"

    def logout(self, values):
        delete_secret("myservice")
        return "Logged out"
```

Field kinds are `text`, `secret`, `bool`, `choice` (with `choices`), `multiline` and `int`.
`label` defaults to the name, `help` is shown under the field and `default` fills a field that
`load()` leaves out.

`save(values)` gets every field by name. It checks them, stores them and returns a short
message, which the window's status bar shows. Raising `ValueError` keeps the dialog open with
your message under the fields, so the user can fix the entry; any other exception is shown as a
failure. Each entry in `actions` adds a button (label to a method name or to `f(values)`) that
returns a message and leaves the dialog open: Test connection, Log out.

Secrets never come back into the dialog. `load()` returns a true value when one is saved (the
field then reads "saved; type to replace") and `save()` gets `""` when the user left it empty,
meaning keep the saved one. Keep secrets in `ewald.keychain`, never in your own file, settings.json
or `os.environ`.

In a package, register the class under the `ewald.settings_pages` entry point group instead of
decorating it.
