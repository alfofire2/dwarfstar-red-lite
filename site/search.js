/* Site search: ⌘K, Ctrl+K or "/" opens it; search.json (scripts/dev/build_site.py) holds one entry per doc section. */
(() => {
  "use strict";
  const root = new URL(".", document.currentScript.src).href;
  const css = `
  .search-btn { display: inline-flex; align-items: center; gap: 10px; padding: 8px 10px 8px 14px; border-radius: 8px;
    border: 1px solid #2E3758; background: rgba(255,255,255,0.02); color: #9AA1B9; font: 13px ui-monospace, "SF Mono", Menlo, monospace;
    cursor: pointer; }
  .search-btn:hover { color: #E9EBF4; border-color: #687090; }
  .search-btn kbd { font: inherit; font-size: 11px; padding: 1px 6px; border: 1px solid #2E3758; border-radius: 5px; color: #687090; }
  dialog.srch { width: min(680px, calc(100vw - 32px)); max-height: min(70vh, 640px); margin: 12vh auto auto; padding: 0;
    border: 1px solid #2E3758; border-radius: 14px; background: #0E1222; color: #E9EBF4; box-shadow: 0 40px 120px rgba(0,0,0,0.6);
    display: none; flex-direction: column; overflow: hidden; }
  dialog.srch[open] { display: flex; }
  dialog.srch::backdrop { background: rgba(5, 7, 14, 0.7); backdrop-filter: blur(3px); }
  .srch-top { display: flex; align-items: center; gap: 12px; padding: 14px 16px; border-bottom: 1px solid #222A45;
    font: 11.5px ui-monospace, "SF Mono", Menlo, monospace; letter-spacing: 1.4px; color: #687090; }
  .srch-top b { color: #FFB070; font-weight: 500; }
  .srch-top input { flex: 1; min-width: 0; background: none; border: 0; outline: 0; color: #E9EBF4;
    font: 17px -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; letter-spacing: 0; }
  .srch-top input::-webkit-search-cancel-button { display: none; }
  .srch-top button { font: inherit; color: #687090; background: none; border: 1px solid #2E3758; border-radius: 5px; padding: 2px 7px; cursor: pointer; }
  .srch-list { list-style: none; margin: 0; padding: 6px; overflow-y: auto; }
  .srch-list a { display: block; padding: 10px 12px; border-radius: 8px; text-decoration: none; color: #9AA1B9; font-size: 14px; line-height: 1.45; }
  .srch-list a[aria-selected="true"] { background: rgba(242, 84, 45, 0.12); outline: 1px solid rgba(242, 84, 45, 0.45); }
  .srch-list .h { display: block; color: #E9EBF4; font-weight: 600; font-size: 15px; }
  .srch-list .t { color: #687090; font: 11px ui-monospace, "SF Mono", Menlo, monospace; letter-spacing: 1.2px; text-transform: uppercase; }
  .srch-list mark { background: none; color: #FF7A52; font-weight: 600; }
  .srch-foot { padding: 10px 16px; border-top: 1px solid #222A45; font: 11px ui-monospace, "SF Mono", Menlo, monospace;
    letter-spacing: 1.4px; color: #687090; display: flex; justify-content: space-between; gap: 12px; }
  .srch-empty { padding: 26px 18px; color: #9AA1B9; font-size: 14.5px; }
  @media (max-width: 480px) { .search-btn kbd { display: none; } dialog.srch { margin-top: 64px; max-height: calc(100vh - 90px); } }`;
  const style = document.createElement("style");
  style.textContent = css;
  document.head.appendChild(style);

  const dlg = document.createElement("dialog");
  dlg.className = "srch";
  dlg.setAttribute("aria-label", "Search the docs");
  dlg.innerHTML = `<div class="srch-top"><b>SEARCH</b><input type="search" placeholder="A flag, a model, a number…"
    aria-label="Search the docs" aria-controls="srch-list" autocomplete="off" spellcheck="false"><button type="button" data-close>ESC</button></div>
    <ul class="srch-list" id="srch-list" role="listbox"></ul>
    <div class="srch-foot"><span data-count>DOCS, FINDINGS, MILESTONES</span><span>↑↓ MOVE · ↵ OPEN</span></div>`;
  document.body.appendChild(dlg);
  const input = dlg.querySelector("input"), list = dlg.querySelector(".srch-list"), count = dlg.querySelector("[data-count]");

  let index = null, hits = [], sel = 0;
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  const reEsc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

  async function load() {
    if (index) return;
    try {
      const r = await fetch(root + "search.json");
      index = (await r.json()).map((e) => ({ ...e, lt: e.t.toLowerCase(), lh: e.h.toLowerCase(), lx: e.x.toLowerCase() }));
    } catch {
      index = [];
      list.innerHTML = '<li class="srch-empty">The search index did not load. Check the connection and open search again.</li>';
    }
  }

  function snippet(x, terms) {
    const lx = x.toLowerCase();
    let at = Math.min(...terms.map((t) => { const i = lx.indexOf(t); return i < 0 ? Infinity : i; }));
    if (!isFinite(at)) at = 0;
    const from = Math.max(0, at - 70), s = (from ? "…" : "") + x.slice(from, from + 210) + (from + 210 < x.length ? "…" : "");
    const re = new RegExp("(" + terms.map(reEsc).join("|") + ")", "gi");
    return esc(s).replace(re, "<mark>$1</mark>");
  }

  function search() {
    const q = input.value.trim().toLowerCase();
    if (!q || !index) { list.innerHTML = ""; count.textContent = "DOCS, FINDINGS, MILESTONES"; hits = []; return; }
    const terms = q.split(/\s+/).filter(Boolean);
    hits = [];
    for (const e of index) {
      let score = 0, ok = true;
      for (const t of terms) {
        const inH = e.lh.includes(t), inT = e.lt.includes(t);
        let n = 0, i = e.lx.indexOf(t);
        while (i >= 0 && n < 6) { n++; i = e.lx.indexOf(t, i + t.length); }
        if (!inH && !inT && !n) { ok = false; break; }
        score += (inH ? 12 : 0) + (inT ? 6 : 0) + n;
      }
      if (ok && terms.length > 1 && (e.lh + " " + e.lx).includes(q)) score += 20;   /* the exact phrase */
      if (ok) hits.push([score, e]);
    }
    hits.sort((a, b) => b[0] - a[0]);
    hits = hits.slice(0, 30).map((h) => h[1]);
    sel = 0;
    count.textContent = hits.length + (hits.length === 30 ? "+" : "") + " HITS";
    list.innerHTML = hits.length
      ? hits.map((e, i) => `<li><a role="option" id="srch-${i}" href="${esc(root + e.u)}" aria-selected="${i === 0}">
          <span class="t">${esc(e.t)}</span><span class="h">${esc(e.h)}</span>${snippet(e.x, terms)}</a></li>`).join("")
      : `<li class="srch-empty">Nothing found for “${esc(input.value.trim())}”. Try a model name, a command such as
          <code>redlite serve</code>, or a number such as 24 GiB.</li>`;
    input.setAttribute("aria-activedescendant", hits.length ? "srch-0" : "");
  }

  function move(d) {
    if (!hits.length) return;
    const items = list.querySelectorAll("a");
    items[sel].setAttribute("aria-selected", "false");
    sel = (sel + d + hits.length) % hits.length;
    items[sel].setAttribute("aria-selected", "true");
    items[sel].scrollIntoView({ block: "nearest" });
    input.setAttribute("aria-activedescendant", "srch-" + sel);
  }

  async function open() {
    if (dlg.open) return;
    dlg.showModal();
    input.select();
    await load();
    search();
  }

  input.addEventListener("input", search);
  input.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { e.preventDefault(); move(1); }
    else if (e.key === "ArrowUp") { e.preventDefault(); move(-1); }
    else if (e.key === "Enter" && hits.length) { e.preventDefault(); list.querySelectorAll("a")[sel].click(); }
  });
  list.addEventListener("click", (e) => { if (e.target.closest("a")) dlg.close(); });
  dlg.querySelector("[data-close]").addEventListener("click", () => dlg.close());
  dlg.addEventListener("click", (e) => { if (e.target === dlg) dlg.close(); });
  document.addEventListener("keydown", (e) => {
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName) || document.activeElement.isContentEditable;
    if ((e.key === "k" && (e.metaKey || e.ctrlKey)) || (e.key === "/" && !typing)) { e.preventDefault(); open(); }
  });
  document.querySelectorAll("[data-search]").forEach((b) => b.addEventListener("click", open));
  if (!/Mac|iPhone|iPad/.test(navigator.platform)) document.querySelectorAll(".search-btn kbd").forEach((k) => (k.textContent = "Ctrl K"));
})();
