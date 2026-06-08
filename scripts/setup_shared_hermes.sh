#!/bin/bash
# Setup shared .hermes on this machine
# Run once per machine to enable shared storage

SHARED_HERMES="/media/scott/S/shared-hermes"
HERMES_HOME=~/.hermes

echo "Setting up shared .hermes..."

# Create ~/.hermes if needed
mkdir -p $HERMES_HOME

# Remove existing symlinks
rm -f $HERMES_HOME/chats $HERMES_HOME/skills $HERMES_HOME/config

# Create new symlinks to shared location
ln -sf $SHARED_HERMES/chats $HERMES_HOME/chats
ln -sf $SHARED_HERMES/skills $HERMES_HOME/skills  
ln -sf $SHARED_HERMES/config $HERMES_HOME/config

# Verify setup
echo "Symlinks created:"
ls -la $HERMES_HOME | grep shared-hermes

echo ""
echo "Shared .hermes setup complete!"
echo "Data location: $SHARED_HERMES"
