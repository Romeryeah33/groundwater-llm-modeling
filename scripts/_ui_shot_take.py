# -*- coding: utf-8 -*-
"""配套 scripts/_ui_shot_make.py：用 CDP 驱动 headless Chrome 给离线 UI 截图。

_ui_shot_make.py 只负责把服务起起来（它不能切前端 tab），本脚本负责连上去、
切 tab、滚动、截图。用 Chrome 自带的 DevTools Protocol，不依赖 selenium /
playwright（这两个环境里没装）；只用到已装的 websockets。

用法（先在另一个终端跑 _ui_shot_make.py，等它打印 READY）：

    python scripts/_ui_shot_take.py --probe            # 列出页面上的按钮/文本，摸结构
    python scripts/_ui_shot_take.py --click 过程分析 --out /tmp/a.png
    python scripts/_ui_shot_take.py                    # 按脚本里的 TARGETS 批量出图

**绝不动用户的 Chrome profile**：每次都用一次性临时 user-data-dir。
"""
import argparse
import asyncio
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parent.parent
CHROME = os.environ.get("CHROME_EXE",
                        r"C:\Program Files\Google\Chrome\Application\chrome.exe")
SITE = f"http://127.0.0.1:{os.environ.get('SHOT_PORT', '7999')}/"
CDP_PORT = int(os.environ.get("CDP_PORT", "9222"))
W, H = 2549, 1254          # 与 9-21 那批截图同尺寸（2549x1254）


class CDP:
    def __init__(self, ws):
        self.ws = ws
        self._i = 0
        self.events = []          # 收集 console / 网络事件，供 --console 打印

    async def call(self, method, **params):
        self._i += 1
        mid = self._i
        await self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"CDP {method} 失败: {msg['error']}")
                return msg.get("result", {})
            elif "method" in msg:
                self.events.append(msg)


async def js(cdp, expr):
    r = await cdp.call("Runtime.evaluate", expression=expr,
                       returnByValue=True, awaitPromise=True)
    res = r.get("result", {})
    if res.get("subtype") == "error":
        raise RuntimeError(f"JS 报错: {res.get('description')}")
    return res.get("value")


async def shot(cdp, path, clip=None, beyond=False):
    p = {"format": "png", "captureBeyondViewport": beyond}
    if clip:
        p["clip"] = {"x": clip[0], "y": clip[1], "width": clip[2],
                     "height": clip[3], "scale": 1}
    r = await cdp.call("Page.captureScreenshot", **p)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(base64.b64decode(r["data"]))
    print(f"  已保存 {path}  ({Path(path).stat().st_size // 1024} KB)", flush=True)


async def click_by_text(cdp, text):
    """点掉文本恰好等于 text 的第一个可见 button（Gradio 的 tab 就是 button）。"""
    ok = await js(cdp, """(() => {
        const t = %s;
        let b = [...document.querySelectorAll('[role=tab]')]
            .filter(e => e.textContent.trim() === t)[0];
        if (!b) b = [...document.querySelectorAll('button')]
            .filter(e => e.textContent.trim() === t && e.offsetParent !== null)[0];
        if (!b) return false;
        b.click();
        return true;
    })()""" % json.dumps(text))
    if not ok:
        raise RuntimeError(f"没找到文本为 {text!r} 的可点 tab")
    print(f"  已点击 tab「{text}」", flush=True)


async def connect(cdp_port, timeout=40, headed=False):
    """起 Chrome、拿 page target 的 ws 地址、开 Page/Runtime 域。

    headed=True 时不开 headless：Gradio 的 Plot 靠 ResizeObserver 拿尺寸，
    headless 下量出来是 0，图会一直卡在 "measuring" 画不出来。带界面跑时
    把窗口挪到屏幕外，免得打扰用户。
    """
    prof = tempfile.mkdtemp(prefix="cdp_shot_")
    args = [CHROME, f"--remote-debugging-port={cdp_port}",
            f"--user-data-dir={prof}", f"--window-size={W},{H}",
            "--no-first-run", "--no-default-browser-check",
            "--hide-scrollbars", "--force-device-scale-factor=1"]
    if headed:
        args += ["--window-position=-3000,-3000"]
    else:
        args += ["--headless=new", "--disable-gpu"]
    args.append("about:blank")
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    ws_url = None
    for _ in range(int(timeout * 2)):
        try:
            raw = urllib.request.urlopen(
                f"http://127.0.0.1:{cdp_port}/json/list", timeout=1).read()
            pages = [t for t in json.loads(raw) if t.get("type") == "page"]
            if pages:
                ws_url = pages[0]["webSocketDebuggerUrl"]
                break
        except Exception:
            pass
        time.sleep(0.5)
    if not ws_url:
        proc.terminate()
        raise RuntimeError("Chrome 的 CDP 端口没起来")
    ws = await websockets.connect(ws_url, max_size=None)   # 截图 base64 会很大
    cdp = CDP(ws)
    await cdp.call("Page.enable")
    await cdp.call("Runtime.enable")
    await cdp.call("Log.enable")
    await cdp.call("Network.enable")
    await cdp.call("Emulation.setDeviceMetricsOverride",
                   width=W, height=H, deviceScaleFactor=1, mobile=False)
    return proc, ws, cdp


async def open_page(cdp, wait=10.0):
    await cdp.call("Page.navigate", url=SITE)
    for _ in range(int(wait * 2)):
        await asyncio.sleep(0.5)
        if await js(cdp, "document.readyState") == "complete":
            break
    await asyncio.sleep(wait)      # Gradio 要再渲染一会儿


async def do_probe(cdp, wait=10.0, click=None):
    await open_page(cdp, wait)
    if click:
        await click_by_text(cdp, click)
        await asyncio.sleep(4.0)
    print("\n=== 页面标题 ===")
    print(await js(cdp, "document.title"))
    print("\n=== 可见 button 文本 ===")
    btns = await js(cdp, """JSON.stringify(
        [...new Set([...document.querySelectorAll('button')]
            .filter(e => e.offsetParent !== null)
            .map(e => e.textContent.trim())
            .filter(t => t.length > 0 && t.length < 30))])""")
    for b in json.loads(btns or "[]"):
        print(f"  [{b}]")
    print("\n=== 可见 tab 类元素 ===")
    tabs = await js(cdp, """JSON.stringify(
        [...document.querySelectorAll('[role=tab], .tab-nav button')]
            .filter(e => e.offsetParent !== null)
            .map(e => e.textContent.trim()))""")
    print(" ", tabs)
    print("\n=== 正文前 600 字 ===")
    txt = await js(cdp, "document.body.innerText.slice(0,600)")
    print(txt)
    print("\n=== 页面高度 ===")
    print(await js(cdp, "document.body.scrollHeight"))
    print("\n=== 已渲染的图片（naturalWidth=0 即没画出来）===")
    print(await js(cdp, """JSON.stringify([...document.images].map(i => ({
        w: i.naturalWidth, h: i.naturalHeight,
        src: i.src.slice(0, 34), cls: i.className.slice(0, 30)})))"""))
    print("\n=== 仍在 loading 的元素数 ===")
    print(await js(cdp, """document.querySelectorAll(
        '.loading, .progress-text, .generating, .wrap.loading, .svelte-loading').length"""))
    print("\n=== 有没有 .gw-plot 容器 / 里面有什么 ===")
    print(await js(cdp, """(() => {
        const e = [...document.querySelectorAll('.gw-plot')];
        return JSON.stringify(e.map(n => ({
            n: n.tagName, html: n.innerHTML.slice(0, 120)})));
    })()"""))


async def amain():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="只列出页面结构")
    ap.add_argument("--click", help="点这个文本的 tab")
    ap.add_argument("--scroll", type=int, default=0, help="点完后滚动到 y")
    ap.add_argument("--out", help="输出 png")
    ap.add_argument("--wait", type=float, default=10.0)
    ap.add_argument("--full", action="store_true", help="截整页长图而非视口")
    ap.add_argument("--sect", help="滚动到这个文本的元素处再截（比 --scroll 稳）")
    ap.add_argument("--tabtop", action="store_true",
                    help="点完 tab 后把该 tab 按钮行滚到距视口顶 60px")
    ap.add_argument("--waitjs", help="轮询直到这段 JS 返回真（最多 180s）")
    ap.add_argument("--console", action="store_true", help="打印浏览器 console/网络失败")
    ap.add_argument("--headed", action="store_true", help="不开 headless（Plot 才画得出来）")
    a = ap.parse_args()

    if not Path(CHROME).exists():
        raise SystemExit(f"找不到 Chrome: {CHROME}（可用 CHROME_EXE 覆盖）")

    proc, ws, cdp = await connect(CDP_PORT, headed=a.headed)
    try:
        if a.probe:
            await do_probe(cdp, a.wait, a.click)
            return
        await open_page(cdp, a.wait)
        if a.click:
            await click_by_text(cdp, a.click)
            await asyncio.sleep(3.0)
        if a.tabtop:
            ok = await js(cdp, """(() => {
                const t = %s;
                let b = [...document.querySelectorAll('[role=tab]')]
                    .filter(e => e.textContent.trim() === t)[0];
                if (!b) b = [...document.querySelectorAll('button')]
                    .filter(e => e.textContent.trim() === t && e.offsetParent !== null)[0];
                if (!b) return false;
                window.scrollTo(0, b.getBoundingClientRect().top + window.scrollY - 60);
                return true;
            })()""" % json.dumps(a.click or ""))
            if not ok:
                raise RuntimeError(f"--tabtop 找不到按钮 {a.click!r}")
            print("  已把 tab 行对齐到视口顶", flush=True)
            await asyncio.sleep(2.0)
        if a.sect:
            ok = await js(cdp, """(() => {
                const t = %s;
                const e = [...document.querySelectorAll('*')]
                    .filter(n => n.children.length === 0 && n.textContent.trim() === t)[0];
                if (!e) return false;
                e.scrollIntoView({block: 'start'});
                return true;
            })()""" % json.dumps(a.sect))
            if not ok:
                raise RuntimeError(f"没找到文本为 {a.sect!r} 的元素")
            print(f"  已滚动到「{a.sect}」", flush=True)
            await asyncio.sleep(1.5)
        if a.scroll:
            await js(cdp, f"window.scrollTo(0,{a.scroll})")
            await asyncio.sleep(1.5)
        if a.waitjs:
            for i in range(180):
                try:
                    if await js(cdp, a.waitjs):
                        print(f"  --waitjs 第 {i}s 成立", flush=True)
                        break
                except Exception:
                    pass
                await asyncio.sleep(1)
            else:
                print("  !! --waitjs 等了 180s 仍不成立", flush=True)
            await asyncio.sleep(3)
        if a.out:
            if a.full:
                m = await cdp.call("Page.getLayoutMetrics")
                cs = m.get("cssContentSize") or m["contentSize"]
                fw, fh = int(cs["width"]), int(cs["height"])
                print(f"  整页 {fw}x{fh}", flush=True)
                await shot(cdp, a.out, clip=(0, 0, fw, min(fh, 20000)), beyond=True)
            else:
                await shot(cdp, a.out)
        if a.console:
            print("\n=== 浏览器事件 ===")
            n = 0
            for e in cdp.events:
                m = e.get("method", "")
                p = e.get("params", {})
                if m == "Runtime.consoleAPICalled":
                    vals = [str(x.get("value", x.get("description", "")))[:300]
                            for x in p.get("args", [])]
                    print(f"  [console.{p.get('type')}] {' | '.join(vals)}")
                    n += 1
                elif m == "Runtime.exceptionThrown":
                    d = p.get("exceptionDetails", {})
                    print(f"  [异常] {d.get('text')} {str(d.get('exception', {}))[:300]}")
                    n += 1
                elif m == "Log.entryAdded":
                    ent = p.get("entry", {})
                    if ent.get("level") in ("error", "warning"):
                        print(f"  [log.{ent.get('level')}] {str(ent.get('text'))[:300]}")
                        n += 1
                elif m == "Network.loadingFailed":
                    print(f"  [网络失败] {p.get('type')} {p.get('errorText')} "
                          f"{str(p.get('requestId'))[:20]}")
                    n += 1
            if not n:
                print("  （没有 console 输出/错误）")
    finally:
        await ws.close()
        proc.terminate()


if __name__ == "__main__":
    asyncio.run(amain())
