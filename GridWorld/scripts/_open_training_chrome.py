"""Open the training UI in Chrome once the local server is ready."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request
import winreg


def chrome_path():
    for root in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            with winreg.OpenKey(root, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\chrome.exe") as key:
                return winreg.QueryValueEx(key, None)[0]
        except OSError:
            pass
    for name in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
        path = Path(os.environ.get(name, "")) / "Google/Chrome/Application/chrome.exe"
        if path.is_file():
            return str(path)
    return shutil.which("chrome")


def training_tab_exists():
    # Inspect Chrome's tab controls, including tabs in the background.
    script = r"""
Add-Type -AssemblyName UIAutomationClient
$root = [System.Windows.Automation.AutomationElement]::RootElement
$windows = $root.FindAll([System.Windows.Automation.TreeScope]::Children,
    [System.Windows.Automation.Condition]::TrueCondition)
foreach ($window in $windows) {
    if ($window.Current.ClassName -ne 'Chrome_WidgetWin_1') { continue }
    $condition = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
        [System.Windows.Automation.ControlType]::TabItem)
    foreach ($tab in $window.FindAll([System.Windows.Automation.TreeScope]::Descendants, $condition)) {
        if ($tab.Current.Name -eq 'AI学習 | GridWorld Lab') { exit 0 }
    }
}
exit 1
"""
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def main(url):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if b"GridWorld Lab" in response.read():
                    if training_tab_exists():
                        return
                    chrome = chrome_path()
                    if chrome:
                        subprocess.Popen([chrome, url], close_fds=True)
                    else:
                        print("Google Chrome was not found. Open " + url + " manually.", flush=True)
                    return
        except (OSError, ValueError):
            pass
        time.sleep(0.5)
    print("The training UI did not become ready: " + url, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
