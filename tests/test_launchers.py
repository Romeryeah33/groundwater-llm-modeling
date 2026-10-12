# -*- coding: utf-8 -*-
"""启动脚本（.bat）的卫生检查 —— 一次真实事故换来的锁。

**事故经过（2026-10-12）**：给 `启动程序.bat` 加了两行中文注释。用户双击后得到：

    '.' 不是内部或外部命令
    't' 不是内部或外部命令
    'packages' 不是内部或外部命令
    'lopy' 不是内部或外部命令          <- 这行来自含 "flopy" 的那句
    'CANDIDATES' 不是内部或外部命令

**根因**：cmd.exe 按**系统代码页**读批处理文件（中文 Windows 上是 GBK），
**不是**按 UTF-8。UTF-8 的中文被当 GBK 解码成别的字节，其中某些字节对会被
解析器当成特殊字符，把**整个文件**的解析带偏 —— 所以报错的行号、内容全对不上，
`flopy` 被断成 `lopy` 这种诡异现象就是这么来的。原文件**一个非 ASCII 字节都没有**，
这个不变量是它一直好用、且好改的原因。

**同类第二个坑**：中文**文件名**也不行。批处理里 `call "%~dp0启动程序.bat"`
实测失败（同样是被错误解码）。所以中文名的 .bat 只能当"双击入口"，
**绝不能被别的脚本引用**；真正的逻辑放在 ASCII 名的 `scripts/launch.bat`。

**第三个坑**：行尾必须是 CRLF。LF 单独存在时，`for ... do ( ... )` 块与
`goto :label` 的解析会出错。

本文件把这三条锁住。脚本风格，不用 pytest：``python tests/test_launchers.py``
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_passed = 0


def _ok(msg):
    global _passed
    _passed += 1
    print(f"   OK: {msg}")


def _bat_files():
    """仓库里所有 .bat（跳过依赖目录）。"""
    skip = {"__pycache__", ".git", "node_modules", ".venv", "venv"}
    return sorted(p for p in ROOT.rglob("*.bat")
                  if not any(part in skip for part in p.parts))


_bats = _bat_files()
assert _bats, "一个 .bat 都没找到 —— 路径写错了？（否则这条检查等于没跑）"
_ok(f"找到 {len(_bats)} 个启动脚本: {[p.name for p in _bats]}")

# ================================================================
print("\n1. 中文只能出现在 chcp 65001 之后（否则纯 ASCII）...")
# ================================================================
# 这不是"不许写中文"，而是**中文必须排在 chcp 65001 后面**。
# 两种都合规的做法：
#   (a) 全文件纯 ASCII —— 最省事，不会出错；
#   (b) 开头先 `chcp 65001`，中文一律排它后面 —— 这样 cmd 读后续字节时
#       已经切到 UTF-8，不会再拿 GBK 去解。
# 不合规的是第三种：**中文排在 chcp 之前**（哪怕是 REM 注释）—— 那几行
# 仍会被 GBK 错读，正是这次把整个文件解析带偏的原因。
for p in _bats:
    raw = p.read_bytes()
    first_cn = next((i for i, b in enumerate(raw) if b > 127), None)
    if first_cn is None:
        continue                                  # 做法 (a)，最安全
    # 必须匹配**一整行命令**（行首 chcp），不能是散文里提到的 `chcp 65001`。
    # 教训：第一版写成子串查找，而文件头那段英文说明里恰好有 "`chcp 65001`"
    # 这几个字符，于是变异测试（把中文挪到 chcp 之前）依然全绿 —— 抓了个寂寞。
    # latin-1 解码是字节保真的，不会因为解码器报错而漏检。
    _head = raw[:first_cn].decode("latin-1")
    _line = raw[:first_cn].count(b"\n") + 1          # 先算好：f-string 里不能出现反斜杠
    assert re.search(r"(?mi)^[ \t]*chcp[ \t]+65001\b", _head), (
        f"{p.relative_to(ROOT)} 第 {_line} 行就出现了非 ASCII，"
        "但在它**之前**没有真正的 `chcp 65001` 命令（注释里提到不算）。\n"
        "cmd.exe 按系统代码页(GBK)读批处理，UTF-8 中文会被解成乱码，"
        "其中某些字节对会把整个文件的解析带偏（症状是 'flopy' 被断成 'lopy' 这种）。\n"
        "两条路：改成纯 ASCII，或把 `chcp 65001 > nul` 提到所有中文之前。")
_ok(f"{len(_bats)} 个脚本要么纯 ASCII，要么中文全在 chcp 65001 之后")

# ================================================================
print("\n2. 行尾必须是 CRLF...")
# ================================================================
for p in _bats:
    raw = p.read_bytes()
    lone_lf = raw.count(b"\n") - raw.count(b"\r\n")
    assert lone_lf == 0, (
        f"{p.relative_to(ROOT)} 有 {lone_lf} 处纯 LF 行尾 —— "
        "cmd 解析 for(...) 块与 goto 标签时会出错，必须全 CRLF")
_ok(f"{len(_bats)} 个脚本行尾全部是 CRLF（没有裸 LF）")

# ================================================================
print("\n3. 批处理里不许引用中文文件名...")
# ================================================================
# 中文名的 .bat 只能当双击入口；一旦被别的脚本 call/start，就会失败
# （实测过：'启动程序.bat' 不是内部或外部命令）。
#
# 实现要点：**在字节上查，不要解码**。第一版把文件按 errors="replace"
# 解成 ascii，UTF-8 的中文全变成 U+FFFD，于是 `[一-鿿]` 永远匹配不到 ——
# 这条断言从来没可能变红。做法改成：从每个 `.bat` 出现处向左回溯出文件名
# 与路径（token），token 里只要有一个 >127 的字节，就是引用了中文名。
_STOP = b" \t\r\n\"'=("
for p in _bats:
    raw = p.read_bytes()
    for m in re.finditer(rb"\.bat\b", raw, re.I):
        j = m.start()
        i = j
        while i > 0 and raw[i - 1] not in _STOP:
            i -= 1
        token = raw[i:j + 4]
        if any(b > 127 for b in token):
            line_no = raw[:i].count(b"\n") + 1
            raise AssertionError(
                f"{p.relative_to(ROOT)} 第 {line_no} 行引用了非 ASCII 的脚本名: {token!r}\n"
                "批处理里引用中文文件名实测会失败（被系统代码页解码成乱码）。"
                "逻辑请放 scripts\\launch.bat（ASCII 名），中文名只做双击入口。")
_ok("没有任何脚本引用中文文件名")

# 正向确认：共享逻辑确实在 ASCII 名的文件里，且两个入口都指向它
_shared = ROOT / "scripts" / "launch.bat"
assert _shared.exists(), "scripts/launch.bat 不见了 —— 两个入口都靠它"
_ok("共享逻辑在 scripts/launch.bat（ASCII 路径）")

_entry_count = 0
for p in _bats:
    if p.parent != ROOT:
        continue
    text = p.read_bytes().decode("ascii", errors="replace").lower()
    if "scripts\\launch.bat" in text or "scripts/launch.bat" in text:
        _entry_count += 1
assert _entry_count >= 2, (
    f"只有 {_entry_count} 个根目录入口指向共享脚本，应当有 2 个"
    "（启动程序.bat 与 启动程序-调试模式.bat）")
_ok(f"根目录 2 个双击入口都 call 同一个共享脚本（不复制逻辑，避免改一处漏一处）")

# 调试入口必须真的把开关设上，否则"调试模式"名不副实
_dbg = ROOT / "启动程序-调试模式.bat"
if _dbg.exists():
    _t = _dbg.read_bytes().decode("ascii", errors="replace")
    assert "GW_SHOW_DEBUG_PANELS=1" in _t, \
        "调试入口没有设 GW_SHOW_DEBUG_PANELS=1 —— 双击后看到的还是普通界面"
    _ok("调试入口确实设了 GW_SHOW_DEBUG_PANELS=1")

# ================================================================
print("\n4. 普通入口不许把调试面板打开...")
# ================================================================
_norm = ROOT / "启动程序.bat"
assert _norm.exists(), "启动程序.bat 不见了"
_t = _norm.read_bytes().decode("ascii", errors="replace")
assert "GW_SHOW_DEBUG_PANELS=1" not in _t, \
    "普通入口把调试面板打开了 —— 分发给同事的那份会串号"
_ok("普通入口不打开调试面板（分发给同事的是这一份）")

# ================================================================
print("\n5. CRLF 这件事写进了仓库（.gitattributes），不靠各机的 autocrlf...")
# ================================================================
# 为什么必须锁这一条：git 默认把 CRLF 归一成 LF 存进索引，签出时是否还原
# 取决于**每台机器的** core.autocrlf。本机是 true 所以看不出问题，但换个
# autocrlf=false 的环境（Linux / CI / 手工关掉的同学）签出来的 .bat 就是裸 LF，
# 恰好触发上面第 2 条防的那个故障。.gitattributes 才能把这条约定带进仓库。
_gattr = ROOT / ".gitattributes"
assert _gattr.exists(), (
    ".gitattributes 不见了。没有它，.bat 的 CRLF 只能靠每台机器的 "
    "core.autocrlf 碰运气；换个环境签出就是裸 LF，启动程序会很难查地失败。")
_gtext = _gattr.read_text(encoding="utf-8")
assert re.search(r"^\s*\*\.bat\s+.*\beol=crlf\b", _gtext, re.M), (
    ".gitattributes 里没有 `*.bat text eol=crlf` —— "
    "批处理的 CRLF 就不再随仓库走，会在别的机器上被签成 LF。")
_ok(".gitattributes 钉住了 `*.bat text eol=crlf`（CRLF 随仓库走）")

# ================================================================
print("\n" + "=" * 60)
print(f"启动脚本: {_passed} 项通过")
print("=" * 60)
