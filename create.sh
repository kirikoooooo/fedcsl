#!/bin/bash

# 检查是否提供了源目录和目标目录参数
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 <source_directory> <target_directory>"
  exit 1
fi

SOURCE_DIR=$1
TARGET_DIR=$2

# 检查源目录是否存在
if [ ! -d "${SOURCE_DIR}" ]; then
  echo "Source directory does not exist: ${SOURCE_DIR}"
  exit 1
fi

# 创建目标目录（如果不存在）
mkdir -p ${TARGET_DIR}

# 获取源目录下的所有子文件夹名，并在目标目录下创建对应的空文件夹
for subdir in $(find ${SOURCE_DIR} -mindepth 1 -maxdepth 1 -type d); do
  # 获取子文件夹的名字（不包含完整路径）
  subdir_name=$(basename ${subdir})
  
  # 在目标目录下创建对应的空文件夹
  if mkdir -p "${TARGET_DIR}/${subdir_name}"; then
    # 如果成功创建了文件夹
    echo "Created empty folder: ${TARGET_DIR}/${subdir_name}"
  else
    # 如果文件夹已经存在或创建失败
    echo "Folder already exists or failed to create: ${TARGET_DIR}/${subdir_name}"
  fi
done

echo "All subdirectories have been processed."