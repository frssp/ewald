; Inno Setup script for the PyInstaller folder dist\ewald.
;   iscc /DAppVersion=0.1.0 /DVariant=cpu /DSourceDir=..\..\dist\ewald packaging\windows\ewald.iss
; Per-user install (no admin), Start menu entries, optional file associations and PATH entry.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef Variant
  #define Variant "cpu"
#endif
#ifndef SourceDir
  #define SourceDir "..\..\dist\ewald"
#endif
#ifndef OutputDir
  #define OutputDir "..\..\dist"
#endif

[Setup]
AppId={{6E3F2B1C-5A0D-4E8B-9C47-3B1E2D0A7F51}
AppName=ewald
AppVersion={#AppVersion}
AppVerName=ewald {#AppVersion} ({#Variant})
AppPublisher=frssp
DefaultDirName={autopf}\ewald
DefaultGroupName=ewald
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
ChangesAssociations=yes
ChangesEnvironment=yes
Compression=lzma2/max
SolidCompression=yes
DiskSpanning=no
OutputDir={#OutputDir}
OutputBaseFilename=ewald-{#AppVersion}-win64-{#Variant}-setup
UninstallDisplayIcon={app}\ewald-gui.exe
SetupIconFile=..\..\src\ewald\assets\icon\ewald.ico
WizardStyle=modern

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked
Name: "assoc"; Description: "Open .lammpstrj and .dump files with ewald"; Flags: unchecked
Name: "addpath"; Description: "Add ewald.exe (command line) to PATH"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\ewald"; Filename: "{app}\ewald-gui.exe"
Name: "{group}\ewald command prompt"; Filename: "{cmd}"; Parameters: "/k set ""PATH={app};%PATH%"" && ewald --help"; WorkingDir: "{userdocs}"
Name: "{group}\Uninstall ewald"; Filename: "{uninstallexe}"
Name: "{autodesktop}\ewald"; Filename: "{app}\ewald-gui.exe"; Tasks: desktopicon

[Registry]
Root: HKA; Subkey: "Software\Classes\ewald.trajectory"; ValueType: string; ValueData: "Atomistic trajectory"; Flags: uninsdeletekey; Tasks: assoc
Root: HKA; Subkey: "Software\Classes\ewald.trajectory\shell\open\command"; ValueType: string; ValueData: """{app}\ewald-gui.exe"" ""%1"""; Tasks: assoc
Root: HKA; Subkey: "Software\Classes\.lammpstrj\OpenWithProgids"; ValueType: string; ValueName: "ewald.trajectory"; ValueData: ""; Flags: uninsdeletevalue; Tasks: assoc
Root: HKA; Subkey: "Software\Classes\.dump\OpenWithProgids"; ValueType: string; ValueName: "ewald.trajectory"; ValueData: ""; Flags: uninsdeletevalue; Tasks: assoc
Root: HKA; Subkey: "Software\Classes\Applications\ewald-gui.exe\SupportedTypes"; ValueType: string; ValueName: ".lammpstrj"; ValueData: ""; Flags: uninsdeletekey; Tasks: assoc

[Run]
Filename: "{app}\ewald-gui.exe"; Description: "Launch ewald"; Flags: nowait postinstall skipifsilent

[Code]
const EnvKey = 'Environment';

function PathRoot: Integer;
begin
  if IsAdminInstallMode then Result := HKLM else Result := HKCU;
end;

function PathKey: String;
begin
  if IsAdminInstallMode then
    Result := 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment'
  else
    Result := EnvKey;
end;

procedure AddToPath(Dir: String);
var Paths: String;
begin
  if not RegQueryStringValue(PathRoot, PathKey, 'Path', Paths) then Paths := '';
  if Pos(';' + Uppercase(Dir) + ';', ';' + Uppercase(Paths) + ';') > 0 then exit;
  if (Paths <> '') and (Copy(Paths, Length(Paths), 1) <> ';') then Paths := Paths + ';';
  RegWriteExpandStringValue(PathRoot, PathKey, 'Path', Paths + Dir);
end;

procedure RemoveFromPath(Dir: String);
var Paths: String; P: Integer;
begin
  if not RegQueryStringValue(PathRoot, PathKey, 'Path', Paths) then exit;
  P := Pos(';' + Uppercase(Dir) + ';', ';' + Uppercase(Paths) + ';');
  if P = 0 then exit;
  if P = 1 then Delete(Paths, 1, Length(Dir) + 1) else Delete(Paths, P - 1, Length(Dir) + 1);
  RegWriteExpandStringValue(PathRoot, PathKey, 'Path', Paths);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if (CurStep = ssPostInstall) and WizardIsTaskSelected('addpath') then
    AddToPath(ExpandConstant('{app}'));
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
    RemoveFromPath(ExpandConstant('{app}'));
end;
