; Installer for the headless service bundle produced by scoresight-service.spec.
;
; The service keeps its configuration under %LOCALAPPDATA%, not next to the
; executable, so the program directory stays read-only as Windows expects.
[Setup]
AppName=ScoreSight Service
AppVersion=@SCORESIGHT_VERSION@
AppPublisher=ScoreSight
DefaultDirName={autopf}\ScoreSight Service
DefaultGroupName=ScoreSight Service
UninstallDisplayName=ScoreSight Service
OutputDir=.\dist
OutputBaseFilename=scoresight-service-setup
Compression=lzma
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64
ArchitecturesAllowed=x64compatible
; Program Files needs elevation; the service itself runs as the logged-in user.
PrivilegesRequired=admin
DisableProgramGroupPage=yes
WizardStyle=modern

[Files]
Source: "dist\scoresight-service\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\ScoreSight Service"; Filename: "{app}\scoresight-service.exe"
Name: "{group}\ScoreSight Service (reachable on the network)"; Filename: "{app}\scoresight-service.exe"; Parameters: "--host 0.0.0.0"
Name: "{group}\Uninstall ScoreSight Service"; Filename: "{uninstallexe}"
Name: "{autodesktop}\ScoreSight Service"; Filename: "{app}\scoresight-service.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked

[Run]
Filename: "{app}\scoresight-service.exe"; Description: "Start ScoreSight Service"; Flags: nowait postinstall skipifsilent
