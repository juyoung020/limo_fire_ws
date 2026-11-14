param($od)
$here=Split-Path $MyInvocation.MyCommand.Path; $bp="C:\Program Files\Bambu Studio\resources\profiles\BBL"
$files=Get-ChildItem "$here\in" -Filter *.stl | % FullName
$a=@('--arrange','1','--orient','0','--load-settings',"$bp\machine\Bambu Lab X2D 0.4 nozzle.json;$here\rig.json",'--load-filaments',"$bp\filament\Bambu PLA Basic @BBL X2D 0.4 nozzle.json",'--slice','0','--outputdir',"$here\$od",'--export-3mf','limo_red1.3mf')+$files
New-Item -ItemType Directory -Force "$here\$od" | Out-Null
$p=Start-Process "C:\Program Files\Bambu Studio\bambu-studio.exe" -ArgumentList ($a | % { if($_ -match ' '){"`"$_`""}else{$_} }) -Wait -PassThru -NoNewWindow
Get-ChildItem "$here\$od" | % Name
foreach($g in Get-ChildItem "$here\$od" -Filter *.gcode){
$t=(Select-String -Path $g.FullName -Pattern "^; model printing time: ([^;]+)").Matches[0].Groups[1].Value
$l=(Select-String -Path $g.FullName -Pattern "^; total filament length \[mm\] : ([\d.]+)").Matches[0].Groups[1].Value
"$($g.Name) time=$t  g=$([math]::Round([double]$l*2.405*1.24/1000,0))" }
Get-Content "$here\$od\result.json" -ErrorAction SilentlyContinue | Select-Object -First 12
