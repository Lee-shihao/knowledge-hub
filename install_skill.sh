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
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
