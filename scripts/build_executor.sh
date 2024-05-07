#!/bin/bash

# Get the directory containing the script file
scriptDirectory="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "The script directory is: $scriptDirectory"

version="0.1"
imageName="clearlovex/taskweaver-executor-pyshark"
imageFullName="$imageName:$version"

taskweaverPath="$scriptDirectory/../taskweaver"
dockerfilePath="$scriptDirectory/../docker/pyshark_container/Dockerfile"
contextPath="$scriptDirectory/../"

if [ -d "$taskweaverPath" ]; then
    echo "Found module files from $taskweaverPath"
    echo "Dockerfile path: $dockerfilePath"
    echo "Context path: $contextPath"
else
    echo "Local files not found."
    exit 1
fi

# Build the Docker image
docker build --no-cache -t "$imageFullName" -f "$dockerfilePath" "$contextPath"

# Tag the image
docker tag "$imageFullName" "$imageName:latest"
