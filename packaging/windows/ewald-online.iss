; Inno Setup script for the online installer (D79): a private Python, then pip installs ewald
; and its dependencies from the network.
;   iscc /DAppVersion=0.3.0 /DPythonDir=..\..\dist\online-python packaging\windows\ewald-online.iss
; PythonDir is the tools\ folder of python.org's NuGet package "python" (a full CPython 3.12 with
; pip); the workflow downloads it. The ewald wheel comes from the release of this version on
; GitHub, everything else from PyPI. /SOURCE=<wheel path or URL> on the setup command line
; installs another wheel (CI passes the one it just built).
; pip in that Python trusts the Windows certificate store (truststore), so proxies that re-sign
; HTTPS work, and it reads the Windows proxy settings. Plugins later install with this same pip
; (plugin_catalog.pip_command; the app is not frozen).

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef PythonDir
  #define PythonDir "..\..\dist\online-python"
#endif
#ifndef OutputDir
  #define OutputDir "..\..\dist"
#endif
#define PyTag "cp312"
#define WheelURL "https://github.com/frssp/ewald/releases/download/v" + AppVersion + "/ewald-" + AppVersion + "-" + PyTag + "-" + PyTag + "-win_amd64.whl"

[Setup]
; not the offline installer's AppId: the two install side by side in different folders
AppId={{B8D4A1E2-7C3F-4F60-9A15-2E6D0C9B4E73}
AppName=ewald
AppVersion={#AppVersion}
AppVerName=ewald {#AppVersion} (online install)
AppPublisher=frssp
DefaultDirName={autopf}\ewald-python
DefaultGroupName=ewald
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
ChangesAssociations=yes
Compression=lzma2/max
SolidCompression=yes
OutputDir={#OutputDir}
OutputBaseFilename=ewald-{#AppVersion}-win64-online-setup
UninstallDisplayIcon={app}\ewald.ico
SetupIconFile=..\..\src\ewald\assets\icon\ewald.ico
WizardStyle=modern

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked
Name: "assoc"; Description: "Open .lammpstrj and .dump files with ewald"; Flags: unchecked
Name: "gpu"; Description: "NVIDIA GPU support (CuPy and the CUDA 12 libraries, about 1.5 GB more to download)"; Flags: unchecked

[Files]
Source: "{#PythonDir}\*"; DestDir: "{app}\python"; Flags: ignoreversion recursesubdirs createallsubdirs
; last file: pip runs after Python is in place and before the shortcuts (which need it) are made
Source: "..\..\src\ewald\assets\icon\ewald.ico"; DestDir: "{app}"; Flags: ignoreversion; AfterInstall: InstallPackages

[Icons]
Name: "{group}\ewald"; Filename: "{app}\python\pythonw.exe"; Parameters: "-m ewald gui"; IconFilename: "{app}\ewald.ico"; WorkingDir: "{userdocs}"; Check: Installed
Name: "{group}\ewald command prompt"; Filename: "{cmd}"; Parameters: "/k set ""PATH={app}\python;{app}\python\Scripts;%PATH%"" && ewald --help"; WorkingDir: "{userdocs}"; Check: Installed
Name: "{group}\Uninstall ewald (online install)"; Filename: "{uninstallexe}"
Name: "{autodesktop}\ewald"; Filename: "{app}\python\pythonw.exe"; Parameters: "-m ewald gui"; IconFilename: "{app}\ewald.ico"; WorkingDir: "{userdocs}"; Tasks: desktopicon; Check: Installed

[Registry]
Root: HKA; Subkey: "Software\Classes\ewald.trajectory"; ValueType: string; ValueData: "Atomistic trajectory"; Flags: uninsdeletekey; Tasks: assoc
Root: HKA; Subkey: "Software\Classes\ewald.trajectory\DefaultIcon"; ValueType: string; ValueData: "{app}\ewald.ico"; Tasks: assoc
Root: HKA; Subkey: "Software\Classes\ewald.trajectory\shell\open\command"; ValueType: string; ValueData: """{app}\python\pythonw.exe"" -m ewald gui ""%1"""; Tasks: assoc
Root: HKA; Subkey: "Software\Classes\.lammpstrj\OpenWithProgids"; ValueType: string; ValueName: "ewald.trajectory"; ValueData: ""; Flags: uninsdeletevalue; Tasks: assoc
Root: HKA; Subkey: "Software\Classes\.dump\OpenWithProgids"; ValueType: string; ValueName: "ewald.trajectory"; ValueData: ""; Flags: uninsdeletevalue; Tasks: assoc

[Run]
Filename: "{app}\python\pythonw.exe"; Parameters: "-m ewald gui"; Description: "Launch ewald"; Flags: nowait postinstall skipifsilent; Check: Installed

[UninstallDelete]
; what pip put there is not in Inno's list of installed files
Type: filesandordirs; Name: "{app}\python"
Type: files; Name: "{app}\install.log"

[Code]
var
  PipOk: Boolean;
  LogPath: String;

function Installed: Boolean;
begin
  Result := PipOk;
end;

{ "ewald[gui] @ <url>"; a local wheel path becomes a file: URL }
function Requirement: String;
var Source, Extras: String;
begin
  Source := ExpandConstant('{param:SOURCE|{#WheelURL}}');
  if Pos('://', Source) = 0 then
  begin
    StringChangeEx(Source, '\', '/', True);
    Source := 'file:///' + Source;
  end;
  Extras := 'gui';
  if WizardIsTaskSelected('gpu') then Extras := 'gui,gpu';
  Result := '"ewald[' + Extras + '] @ ' + Source + '"';
end;

procedure OnPipLine(const S: String; const Error, FirstLine: Boolean);
begin
  SaveStringToFile(LogPath, S + #13#10, True);
  if not WizardSilent then
  begin
    WizardForm.FilenameLabel.Caption := Copy(S, 1, 120);
  end;
end;

function RunPython(const Params: String): Integer;
var Code: Integer;
begin
  SaveStringToFile(LogPath, '> python ' + Params + #13#10, True);
  if not ExecAndLogOutput(ExpandConstant('{app}\python\python.exe'), Params, ExpandConstant('{app}'),
                          SW_HIDE, ewWaitUntilTerminated, Code, @OnPipLine) then
    Code := -1;
  Result := Code;
end;

procedure InstallPackages;
var Extra: String;
begin
  LogPath := ExpandConstant('{app}\install.log');
  DeleteFile(LogPath);
  if not WizardSilent then
    WizardForm.StatusLabel.Caption := 'Downloading and installing ewald and the packages it needs (a few minutes)...';
  if RunPython('-m pip --version') <> 0 then
    RunPython('-m ensurepip --default-pip');
  Extra := '';
  if WizardIsTaskSelected('gpu') then
    Extra := ' "cupy-cuda12x[ctk]"';
  PipOk := RunPython('-m pip install --disable-pip-version-check --no-warn-script-location --progress-bar off '
                     + Requirement + Extra) = 0;
  if PipOk then
    PipOk := RunPython('-c "import ewald, ewald.gui, pymatgen; print(''ewald'', ewald.__version__)"') = 0;
  if not PipOk then
    SuppressibleMsgBox('ewald could not be downloaded and installed. Check the internet connection (and a proxy, '
      + 'if your network uses one), then run this setup again.' + #13#10#13#10
      + 'Details are in ' + LogPath + #13#10#13#10
      + 'Without internet, use the offline installer ewald-{#AppVersion}-win64-cpu-setup.exe from the same release.',
      mbError, MB_OK, IDOK);
end;

function GetCustomSetupExitCode: Integer;
begin
  if PipOk then Result := 0 else Result := 1;
end;
