# -*- coding: utf-8 -*-
"""静态校验 .dockerignore 是否真的挡住了该挡的、又没误伤运行时需要的。

**为什么要手写一个匹配器**
本机没有 docker（`docker: command not found`），`docker build` 跑不了，
也就没法用"构建一次再看里面有没有 .env"来验证。而"排除表写错"的失败模式
是**静态可查的**（写错路径、`*` 跨越了 `/`、`!` 被前面的目录排除吃掉），
所以这里按 Docker 文档公开的语义复刻一遍匹配规则，用来查这类错。

**这不是 docker 本身**：它是"我理解的 docker 规则"。真正的构建验证仍需
在装 docker 的机器上做一次（见 docs/后续开发计划20261010.md 部署待办）。

Docker 的关键语义（与 .gitignore **不同**，容易写错）：
  · 模式是拿**相对于构建上下文根**的路径去全匹配的；
  · `*` **不跨越 `/`** —— 所以 `*.log` 只匹配根目录下的 .log，
    不匹配 `logs/debug.log`（后者靠 `logs/` 这条目录模式挡住）；
  · `**` 可跨任意层目录（含零层）；
  · 模式命中**目录**时，该目录及其全部内容一并排除（不再往里递归）；
  · `!` 例外按**书写顺序**生效，后写的覆盖先写的。
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
IGNORE = ROOT / ".dockerignore"


def _pattern_to_regex(pat: str) -> re.Pattern:
    """把一条 .dockerignore 模式编译成锚定的全匹配正则。"""
    out = []
    i = 0
    n = len(pat)
    while i < n:
        c = pat[i]
        if c == "*":
            if pat.startswith("**/", i):        # **/ 也匹配"零层目录"
                out.append(r"(?:.*/)?")
                i += 3
                continue
            if pat.startswith("**", i):
                out.append(r".*")
                i += 2
                continue
            out.append(r"[^/]*")
            i += 1
            continue
        if c == "?":
            out.append(r"[^/]")
            i += 1
            continue
        if c == "[":
            j = pat.find("]", i + 1)
            if j == -1:
                out.append(re.escape(c))
                i += 1
                continue
            out.append(pat[i:j + 1])           # 字符类原样交给正则
            i = j + 1
            continue
        out.append(re.escape(c))
        i += 1
    return re.compile("^" + "".join(out) + "$")


def _load_rules():
    rules = []
    for raw in IGNORE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        negate = line.startswith("!")
        if negate:
            line = line[1:].strip()
        line = line.rstrip("/")                # docker 会做 Clean，尾斜杠无意义
        if not line:
            continue
        rules.append((negate, _pattern_to_regex(line)))
    return rules


def is_excluded(rel_path: str, rules) -> bool:
    """路径自身**或它的任一祖先目录**命中排除模式，即被排除。"""
    parts = rel_path.split("/")
    candidates = ["/".join(parts[:k]) for k in range(1, len(parts) + 1)]
    excluded = False
    for candidate in candidates:
        for negate, rx in rules:
            if rx.match(candidate):
                excluded = not negate
    return excluded


def main() -> int:
    rules = _load_rules()

    # 必须被挡住 —— 每一条都写明"为什么"
    must_exclude = {
        ".env":                        "真实 LLM API Key",
        ".env.bak.20260916":           "密钥备份，与 .env 同内容",
        "data/sessions/default.json":  "用户真实对话记录",
        "data/uploads/x.shp":          "用户上传的场地数据",
        "data/outputs/figures/a.png":  "运行时产物",
        "data/semantic_db.sqlite":     "由 init_semantic_db.py 重建，不必进层",
        "logs/debug_20260619_100351.log": "调试日志",
        "C:Tempgradio_stderr.log":     "根目录散落日志",
        ".git/config":                 "版本历史（remote URL 可能带凭据）",
        "tests/test_gwt_pipeline.py":  "测试代码，运行时不需要",
        "app/__pycache__/x.cpython-311.pyc": "字节码",
        "deploy/deploy.sh":            "部署脚本，运行时不需要",
        "docker/Dockerfile":           "构建时由 builder 单独读取，不进 COPY",
    }

    # 必须**保留** —— 误伤这些会直接构建失败或运行报错
    must_keep = {
        "app/main.py":                     "程序入口",
        "app/frontend/gradio_app.py":      "界面",
        "app/knowledge/db_init.sql":       "语义库建表数据",
        "app/knowledge/semantic_db.py":    "语义库接口",
        "scripts/init_semantic_db.py":     "构建时 RUN 这一步要执行它",
        "requirements.txt":                "构建时 RUN pip install 要读它",
        ".env.example":                    "模板，不含真实密钥",
        "data/cases/case_a_column.json":   "示例算例（顺带保证 data/ 目录存在）",
        "README.md":                       "说明文件",
    }

    failed = 0
    print("=" * 64)
    print("A. 必须被排除")
    print("=" * 64)
    for path, why in must_exclude.items():
        got = is_excluded(path, rules)
        mark = "✅" if got else "❌ 漏了"
        if not got:
            failed += 1
        print(f"  {mark}  {path:38s} {why}")

    print()
    print("=" * 64)
    print("B. 必须保留（误伤即构建/运行失败）")
    print("=" * 64)
    for path, why in must_keep.items():
        got = is_excluded(path, rules)
        mark = "✅" if not got else "❌ 误伤"
        if got:
            failed += 1
        print(f"  {mark}  {path:38s} {why}")

    print()
    print("=" * 64)
    total = len(must_exclude) + len(must_keep)
    if failed == 0:
        print(f"✅ .dockerignore 静态校验通过：{total} 条全部符合预期")
    else:
        print(f"❌ {failed}/{total} 条不符合预期")
    print("=" * 64)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
