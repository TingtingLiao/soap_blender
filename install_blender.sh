#!/bin/bash

# Install Blender 3.6.22
echo "Installing Blender 3.6.22..."
wget https://ftp.halifax.rwth-aachen.de/blender/release/Blender3.6/blender-3.6.22-linux-x64.tar.xz 

# Extract the archive
echo "Extracting archive..."
tar -xf blender-3.6.22-linux-x64.tar.xz

# Move the extracted files to /opt/blender-3.6
sudo mv blender-3.6.22-linux-x64 /opt/blender-3.6

# Create a symbolic link to blender
sudo ln -s /opt/blender-3.6/blender /usr/local/bin/blender

# Clean up
rm -rf blender-3.6.22-linux-x64.tar.xz

# test blender
blender -b 



# echo "Blender 3.6.22 installed successfully!"