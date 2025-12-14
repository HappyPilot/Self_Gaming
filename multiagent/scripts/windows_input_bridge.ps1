#requires -version 5.1
param([string]$Prefix = 'http://+:5000/')

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
Add-Type -Namespace Win32 -Name NativeMouse -MemberDefinition @"
    [DllImport("user32.dll", SetLastError=true)]
    public static extern void mouse_event(uint dwFlags, uint dx, uint dy, uint dwData, int dwExtraInfo);
"@

$MOUSEEVENTF_MOVE = 0x0001
$MOUSEEVENTF_LEFTDOWN = 0x0002
$MOUSEEVENTF_LEFTUP = 0x0004
$MOUSEEVENTF_RIGHTDOWN = 0x0008
$MOUSEEVENTF_RIGHTUP = 0x0010
$MOUSEEVENTF_MIDDLEDOWN = 0x0020
$MOUSEEVENTF_MIDDLEUP = 0x0040

function Invoke-KeyPress { param([string]$Key) [System.Windows.Forms.SendKeys]::SendWait($Key) }
function Invoke-MouseClick {
    param([string]$Button)
    switch ($Button.ToLower()) {
        'click_primary'   { [Win32.NativeMouse]::mouse_event($MOUSEEVENTF_LEFTDOWN,0,0,0,0); [Win32.NativeMouse]::mouse_event($MOUSEEVENTF_LEFTUP,0,0,0,0) }
        'click_secondary' { [Win32.NativeMouse]::mouse_event($MOUSEEVENTF_RIGHTDOWN,0,0,0,0); [Win32.NativeMouse]::mouse_event($MOUSEEVENTF_RIGHTUP,0,0,0,0) }
        'click_middle'    { [Win32.NativeMouse]::mouse_event($MOUSEEVENTF_MIDDLEDOWN,0,0,0,0); [Win32.NativeMouse]::mouse_event($MOUSEEVENTF_MIDDLEUP,0,0,0,0) }
        default { throw "Unsupported mouse action $Button" }
    }
}
function Invoke-MouseMove {
    param([int]$DX,[int]$DY)
    [Win32.NativeMouse]::mouse_event($MOUSEEVENTF_MOVE, [int]$DX, [int]$DY, 0, 0)
}

$listener = [System.Net.HttpListener]::new()
$listener.Prefixes.Add($Prefix)
$listener.Start()

try {
    while ($listener.IsListening) {
        $context = $listener.GetContext()
        $reader = New-Object IO.StreamReader($context.Request.InputStream, $context.Request.ContentEncoding)
        $body = $reader.ReadToEnd()
        try {
            $data = $null
            if (-not [string]::IsNullOrWhiteSpace($body)) { $data = $body | ConvertFrom-Json }
            if ($null -eq $data) { throw "Empty payload" }
            switch ($data.type) {
                'keyboard'   { Invoke-KeyPress $data.key }
                'mouse'      { Invoke-MouseClick $data.action }
                'mouse_move' { Invoke-MouseMove $data.dx $data.dy }
                default      { throw "Unknown input type $($data.type)" }
            }
            $resp = @{ ok = $true; type = $data.type } | ConvertTo-Json -Compress
            $bytes = [System.Text.Encoding]::UTF8.GetBytes($resp)
            $context.Response.StatusCode = 200
            $context.Response.OutputStream.Write($bytes,0,$bytes.Length)
        }
        catch {
            $context.Response.StatusCode = 500
            $err = @{ ok=$false; error=$_.Exception.Message } | ConvertTo-Json -Compress
            $bytes = [System.Text.Encoding]::UTF8.GetBytes($err)
            $context.Response.OutputStream.Write($bytes,0,$bytes.Length)
        }
        finally { $context.Response.OutputStream.Close() }
    }
}
finally { $listener.Stop(); $listener.Close() }
