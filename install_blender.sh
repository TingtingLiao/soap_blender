# #!/bin/bash

# Install Blender 3.6.22
echo -e "\033[0;32mInstalling Blender 3.6.22...\033[0m"
wget https://ftp.halifax.rwth-aachen.de/blender/release/Blender3.6/blender-3.6.22-linux-x64.tar.xz 

# Extract the archive
echo -e "\033[0;32mExtracting archive...\033[0m"
tar -xf blender-3.6.22-linux-x64.tar.xz

# Move the extracted files to /opt/blender-3.6
sudo mv blender-3.6.22-linux-x64 /opt/blender-3.6

# Create a symbolic link to blender
sudo ln -s /opt/blender-3.6/blender /usr/local/bin/blender

# Clean up
rm -rf blender-3.6.22-linux-x64.tar.xz

# Test if Blender is installed correctly
echo -e "\033[0;32mTesting Blender installation...\033[0m"
blender_output=$(blender -b 2>&1)

# Check if the output contains the expected version
if echo "$blender_output" | grep -q "Blender 3.6.22"; then
    echo -e "\033[0;32m ✓ Blender 3.6.22 installed successfully.\033[0m"
    echo -e "\033[0;32mOutput:\033[0m"
    echo "$blender_output"
else
    echo -e "\033[0;31m ✗ Blender installation failed or version mismatch.\033[0m"
    echo "Output:"
    echo "$blender_output"
    exit 1
fi

echo "Installing requirements..."
/opt/blender-3.6/3.6/python/bin/python3.10 -m pip install --target /opt/blender-3.6/3.6/python/lib/python3.10/site-packages -r requirements.txt