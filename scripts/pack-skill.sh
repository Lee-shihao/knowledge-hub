#!/bin/bash

# 配置
SKILL_DIR="skill"
SKILL_NAME="knowledge-hub" # 安装后的目录名
OUT_SCRIPT="install_skill.sh"

echo "正在打包并生成安装脚本: $OUT_SCRIPT ..."

# 1. 将 skill 目录打包为 tar.gz 并转为 base64
TARBALL_BASE64=$(tar czf - "$SKILL_DIR" | base64)

# 2. 写入脚本头部逻辑
cat << 'EOF' > "$OUT_SCRIPT"
#!/bin/bash
# ==========================================
# Knowledge Hub Skill 自解压安装脚本
# 用法: ./install_skill.sh [hermes|claude|openclaw]
# ==========================================

TARGET_AGENT=$1
SKILL_NAME="knowledge-hub"

case "$TARGET_AGENT" in
    hermes)
        TARGET_DIR="$HOME/.hermes/skills"
        ;;
    claude|claude-code)
        TARGET_DIR="$HOME/.claude/skills"
        ;;
    openclaw)
        TARGET_DIR="$HOME/.openclaw/skills"
        ;;
    *)
        echo "❌ 未知的目标 Agent。"
        echo "👉 用法: $0 [hermes|claude|openclaw]"
        exit 1
        ;;
esac

echo "📦 正在安装 $SKILL_NAME 到 $TARGET_DIR ..."

# 确保目标目录存在
mkdir -p "$TARGET_DIR"

# 创建临时目录
TMP_DIR=$(mktemp -d)

# 提取脚本末尾的 __TARBALL_BEGIN__ 之后的 base64 数据，解码并解压
sed -n '/^__TARBALL_BEGIN__$/,$p' "$0" | tail -n +2 | base64 -d | tar xz -C "$TMP_DIR"

# 检查解压是否成功并移动文件
if [ -d "$TMP_DIR/skill" ]; then
    rm -rf "$TARGET_DIR/$SKILL_NAME"
    cp -r "$TMP_DIR/skill" "$TARGET_DIR/$SKILL_NAME"
    echo "✅ 安装成功！路径: $TARGET_DIR/$SKILL_NAME"
    echo "包含文件:"
    ls -la "$TARGET_DIR/$SKILL_NAME"
else
    echo "❌ 解压失败，请检查脚本完整性。"
fi

# 清理临时文件
rm -rf "$TMP_DIR"
exit 0

__TARBALL_BEGIN__
EOF

# 3. 将 Base64 数据追加到脚本末尾
echo "$TARBALL_BASE64" >> "$OUT_SCRIPT"

# 4. 赋予执行权限
chmod +x "$OUT_SCRIPT"

echo "✅ 生成完毕！你可以直接运行: ./$OUT_SCRIPT hermes"
