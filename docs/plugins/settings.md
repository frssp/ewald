# Settings pages

A plugin that needs setting up (a server address, a login, an access key) can give itself a
settings page, so users do it in the GUI instead of in Python or a terminal. ewald builds the
page from a list of fields. It is a page of **Preferences** (after ewald's own pages, with the
icon of the plugin's [side bar](gui.md) if it has one; Apply and OK save it), and it also opens
as a small dialog from the plugin's **Settings** button in the Plugins side bar and from Edit ›
Plugin settings.

![A plugin's page in Preferences](../images/plugin_settings_page.png)

## Letting ewald store the values

Without `load()` and `save()`, ewald keeps the values itself: plain fields in
`<config dir>/plugin-settings/<name>.json`, secret fields in the system keychain. The plugin
reads them with `plugin_settings(name)`, which gives every field (a secret as the secret itself,
or None):

```python
from ewald.plugin_api import SettingsField, plugin_settings, settings_page


@settings_page("sim-agent", title="Sim agent", fields=[
    SettingsField("server", "Server", default="http://localhost:8000"),
    SettingsField("token", "Access token", kind="secret"),
    SettingsField("model", kind="choice", choices=("default", "fast", "accurate"))])
class SimAgentSettings:
    """The agent behind the Sim agent side bar."""


plugin_settings("sim-agent")   # {"server": "http://localhost:8000", "token": None, "model": "default"}
```

The values stay when the plugin is switched off or removed, so they are back when it returns.

## Storing them yourself

When the values live elsewhere (the plugin's own config file, a service that checks them), give
the class `load()` and `save(values)`:

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
