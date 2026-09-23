' ============================================================
' launch_app.vbs
' 雙擊這個檔案即可開啟吸收光譜擬合工具，完全不會跳出黑色
' 命令視窗。可以把這個檔案的捷徑拉到桌面，當作雙擊圖示使用。
'
' 運作原理：呼叫同資料夾內的 launch_app.bat，並以「隱藏視窗」
' 模式執行 (第二個參數 0 代表隱藏)。
' ============================================================

Set objShell = CreateObject("WScript.Shell")
Set objFSO = CreateObject("Scripting.FileSystemObject")

scriptDir = objFSO.GetParentFolderName(WScript.ScriptFullName)
batPath = scriptDir & "\launch_app.bat"

objShell.Run """" & batPath & """", 0, False
