#!/bin/bash

# 设置 GammaGL 目录的路径
# GAMMAGL_DIR="/home/zgy/GammaGL"
GAMMAGL_DIR="/home/zgy/GammaGL"

# 删除 build、dist 和 gammagl.egg-info 目录
rm -rf "$GAMMAGL_DIR/build" "$GAMMAGL_DIR/dist" "$GAMMAGL_DIR/gammagl.egg-info"

# 删除 GammaGL/gammagl/mpops/torch_ext 路径下以 .so 结尾的文件
find "$GAMMAGL_DIR/gammagl/mpops/torch_ext" -name "*.so" -type f -exec rm -f {} +
find "$GAMMAGL_DIR/gammagl/ops" -name "*.so" -type f -exec rm -f {} +
find "$GAMMAGL_DIR/gammagl/mpops/torch_ext" -name "*.pyd" -type f -exec rm -f {} +
find "$GAMMAGL_DIR/gammagl/ops" -name "*.pyd" -type f -exec rm -f {} +

echo "Cleanup completed."

pip uninstall gammagl
