# Set the current directory and target directory

# Directory containing files
$files_dir = "D:"

# Loop through each file in the directory
Get-ChildItem -Path $files_dir -Filter * | ForEach-Object {
    if ($_ -is [System.IO.FileInfo]) {
        # Extract year, month, and day from the filename
        $filename = $_.Name
        $year = $filename -split ('_')[0]
        $month = $filename.Substring(5, 2)
        $day = $filename.Substring(7, 2)

        # Create directory structure if it doesn't exist
        $dest_dir = Join-Path -Path $files_dir -ChildPath "$year\$month\$day"
        New-Item -ItemType Directory -Path $dest_dir -Force | Out-Null

        # Move the file to the appropriate directory
        Move-Item -Path $_.FullName -Destination $dest_dir -Force
        Write-Host "Moved $($_.FullName) to $dest_dir"
    }
}