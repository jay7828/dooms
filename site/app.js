(function () {
  "use strict";
  const D = window.DOOMS;
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
  const color = (sys) => `var(--s${D.systems[sys].slot})`;
  const fmt = (v) => (v >= 1000 ? v.toLocaleString("en-US", { maximumFractionDigits: 0 }) : v >= 100 ? v.toFixed(1).replace(/\.0$/, "") : v < 10 ? v.toFixed(2) : v.toFixed(1));

  // ---------------------------------------------------------------- reveal on first sight (one-shot; instant under reduced motion)
  const io = "IntersectionObserver" in window && !reduce
    ? new IntersectionObserver((entries) => {
        for (const e of entries) {
          if (!e.isIntersecting) continue;
          e.target.dispatchEvent(new CustomEvent("reveal"));
          io.unobserve(e.target);
        }
      }, { threshold: 0.35 })
    : null;
  function onReveal(el, fn) {
    if (!io) { fn(); return; }
    el.addEventListener("reveal", fn, { once: true });
    io.observe(el);
  }

  // ---------------------------------------------------------------- tooltip
  const tip = $("#tip");
  function showTip(text, x, y) { tip.textContent = text; tip.hidden = false; tip.style.left = x + "px"; tip.style.top = y + "px"; }
  function hideTip() { tip.hidden = true; }

  // ---------------------------------------------------------------- horizontal bars, labeled directly
  // rows: [{sys, name, value}], unit: "ms" | "tokens/s", better: "lower" | "higher"
  function barChart(el, rows, unit, better, opts = {}) {
    el.innerHTML = "";
    const max = Math.max(...rows.map((r) => r.value));
    const best = better === "lower" ? Math.min(...rows.map((r) => r.value)) : max;
    const ref = opts.ref; // value the ratio is computed against (the engine's)
    const bars = [];
    rows.forEach((r, i) => {
      const row = document.createElement("div");
      row.className = "row";
      row.dataset.sys = r.sys;
      const name = document.createElement("div");
      name.className = "row-name";
      name.textContent = r.name;
      const track = document.createElement("div");
      track.className = "track";
      const bar = document.createElement("div");
      bar.className = "bar";
      bar.style.width = `calc((100% - 132px) * ${(r.value / max).toFixed(4)})`;
      bar.style.minWidth = "3px";
      bar.style.background = color(r.sys);
      bar.style.transitionDelay = reduce ? "0s" : `${i * 90}ms`;
      const val = document.createElement("div");
      val.className = "val";
      let ratio = "";
      if (ref && r.sys !== "dooms") {
        const x = better === "lower" ? r.value / ref : ref / r.value;
        ratio = x >= 1.005 ? `${x.toFixed(2)}x slower` : x <= 0.995 ? `${(1 / x).toFixed(2)}x faster` : "level";
      }
      val.innerHTML = `${fmt(r.value)} ${unit}${ratio ? `<small>${ratio}</small>` : ""}`;
      val.style.transitionDelay = reduce ? "0s" : `${0.4 + i * 0.09}s`;
      const hit = document.createElement("div");
      hit.className = "hit";
      const label = `${r.name}: ${fmt(r.value)} ${unit}${r.value === best ? " (best)" : ""}`;
      hit.addEventListener("pointermove", (e) => showTip(label, e.clientX, e.clientY));
      hit.addEventListener("pointerleave", hideTip);
      track.append(bar, val, hit);
      row.append(name, track);
      el.append(row);
      bars.push(bar, val);
    });
    const go = () => bars.forEach((b) => b.classList.add("in"));
    if (opts.immediate || reduce) requestAnimationFrame(() => requestAnimationFrame(go));
    else onReveal(el, go);
  }

  // highlight form: the engine in the Dooms orange, every baseline in one neutral grey, each named on its own row
  function legend(el, systems) {
    const others = systems.filter((s) => s !== "dooms").map((s) => D.systems[s].label);
    el.innerHTML = `<span><i style="background:var(--s1)"></i>dooms.cpp</span>` + (others.length ? `<span><i style="background:var(--s2)"></i>${others.join(", ")}</span>` : "");
  }

  // ---------------------------------------------------------------- hero
  const engineVla = D.vla.totals[0].ms;
  barChart($("#hero-chart"), D.vla.totals.map((t) => ({ sys: t.sys, name: t.name, value: t.ms })), "ms", "lower", { ref: engineVla, immediate: true });

  // ---------------------------------------------------------------- where the time goes: a race in real proportion
  // Both runs start together; each row is uncovered at a constant rate, so the engine's row finishes after 266.6/603.8 of the time PyTorch takes.
  (function timeline() {
    const el = $("#timeline");
    const e2e = { dooms: D.vla.totals[0].ms, pytorch: D.vla.totals.find((t) => t.name === "PyTorch MPS fp32").ms };
    const rows = [["dooms", "dooms.cpp", "Neural Engine"], ["pytorch", "PyTorch", "MPS, fp32"]];
    const total = (s) => D.vla.stages[s].reduce((a, b) => a + b.ms, 0);
    const max = Math.max(total("dooms"), total("pytorch"));
    const parts = {};
    rows.forEach(([key, name, sub]) => {
      const row = document.createElement("div");
      row.className = "tl-row";
      const lab = document.createElement("div");
      lab.className = "tl-label";
      lab.innerHTML = `<strong>${name}</strong>${sub}`;
      const wrap = document.createElement("div");
      const track = document.createElement("div");
      track.className = "tl-track";
      track.style.width = `${(total(key) / max) * 100}%`;
      D.vla.stages[key].forEach((st, i) => {
        const seg = document.createElement("div");
        seg.className = "seg";
        seg.dataset.i = i;
        seg.style.flex = `${st.ms} 0 0`;
        seg.textContent = `${fmt(st.ms)} ms`;
        const label = `${name}, ${st.stage}: ${fmt(st.ms)} ms`;
        seg.addEventListener("pointermove", (e) => showTip(label, e.clientX, e.clientY));
        seg.addEventListener("pointerleave", hideTip);
        track.append(seg);
      });
      const cover = document.createElement("div");
      cover.className = "tl-cover";
      track.append(cover);
      const done = document.createElement("div");
      done.className = "tl-done";
      done.innerHTML = key === "dooms" ? `done in <b>${fmt(e2e[key])} ms</b>` : `done in ${fmt(e2e[key])} ms`;
      wrap.append(track, done);
      row.append(lab, wrap);
      el.append(row);
      parts[key] = { cover, done, ms: e2e[key] };
    });
    $("#stage-key").innerHTML = D.vla.stages.dooms
      .map((s, i) => `<span><i style="background:var(--stage-${i + 1})"></i>${s.stage}</span>`).join("");

    const clock = $("#race-clock");
    const longest = Math.max(e2e.dooms, e2e.pytorch);
    const wall = 3200; // ms of animation for the slower run
    let raf = 0;
    function finish() {
      for (const k in parts) { parts[k].cover.style.transform = "scaleX(0)"; parts[k].done.classList.add("in"); }
      clock.textContent = `${fmt(longest)} ms`;
    }
    function run() {
      cancelAnimationFrame(raf);
      if (reduce) { finish(); return; }
      for (const k in parts) { parts[k].cover.style.transform = "scaleX(1)"; parts[k].done.classList.remove("in"); }
      const t0 = performance.now();
      const tick = (now) => {
        const simulated = Math.min(longest, ((now - t0) / wall) * longest);
        clock.textContent = `${Math.round(simulated)} ms`;
        for (const k in parts) {
          const p = Math.min(1, simulated / parts[k].ms);
          parts[k].cover.style.transform = `scaleX(${1 - p})`;
          if (p >= 1) parts[k].done.classList.add("in");
        }
        if (simulated < longest) raf = requestAnimationFrame(tick);
      };
      raf = requestAnimationFrame(tick);
    }
    if (reduce) finish(); else for (const k in parts) parts[k].cover.style.transform = "scaleX(1)";
    onReveal(el, run);
    $("#replay").addEventListener("click", run);
  })();

  // ---------------------------------------------------------------- benchmark tabs
  const TABS = {
    vla: {
      title: "SmolVLA",
      sub: "three cameras, a prompt and the robot state to a 50-step action chunk",
      charts: [{ title: "milliseconds per observation, lower is better", rows: D.vla.totals.map((t) => ({ sys: t.sys, name: t.name, value: t.ms })), unit: "ms", better: "lower" }],
      note: `<p><strong>2.3x faster than PyTorch</strong> on the GPU and 1.8x faster than MLX, with all three stages on the Neural Engine.</p>
             <p>The action chunk agrees with lerobot's fp32 output to 1.2e-2 relative error on identical inputs and noise. The MLX port differs from lerobot by 0.25 on the same inputs.</p>`,
    },
    vlm: {
      title: "SmolVLM-256M",
      sub: "one 512 by 512 image and a question to an answer",
      charts: [
        { title: "vision tower, ms, lower is better", rows: D.vlm.vision.map((t) => ({ sys: t.sys, name: t.name, value: t.ms })), unit: "ms", better: "lower" },
        { title: "time to first token, ms, lower is better", rows: D.vlm.ttft.map((t) => ({ sys: t.sys, name: t.name, value: t.ms })), unit: "ms", better: "lower" },
        { title: "decode, ms per token, lower is better", rows: D.vlm.decode.map((t) => ({ sys: t.sys, name: t.name, value: t.ms })), unit: "ms", better: "lower" },
      ],
      note: `<p><strong>Fastest decode: 1.27x MLX, 1.5x llama.cpp, 10x PyTorch</strong>, on the engine's own Metal decoder. First token 15 ms ahead of MLX; the engine's figure includes image preprocessing, the others start from prepared tensors.</p>
             <p>Tokens are identical to transformers on every reference photo.</p>`,
    },
    llm: {
      title: "Qwen2.5-0.5B-Instruct",
      sub: "512-token prompt, then 128 greedy tokens",
      charts: [
        { title: "decode, f16, tokens per second, higher is better", rows: D.llm.decode.map((t) => ({ sys: t.sys, name: t.name, value: t.tps })), unit: "tok/s", better: "higher" },
        { title: "prefill, f16, tokens per second, higher is better", rows: D.llm.prefill.map((t) => ({ sys: t.sys, name: t.name, value: t.tps })), unit: "tok/s", better: "higher" },
        { title: "decode at 4-bit (a different precision), tokens per second", rows: D.llm.decode4.map((t) => ({ sys: t.sys, name: t.name, value: t.tps })), unit: "tok/s", better: "higher" },
      ],
      note: () => `${D.llm.provisional ? '<span class="badge">single run, final interleaved run pending</span>' : ""}
             <p><strong>Fastest f16 decode</strong>: the engine's own Metal decoder runs Qwen2 and llama files. The prompt is 13% behind llama.cpp, ahead of MLX and PyTorch.</p>
             <p>Quantized files still run on llama.cpp inside the engine. Answers match transformers token for token, 29 of 29.</p>`,
    },
    speech: {
      title: "Whisper speech to text",
      sub: "481 s of LibriSpeech audio, greedy, f16",
      charts: [
        { title: "large-v3-turbo, 73 clips, seconds, lower is better", rows: D.speech.turboClips.map((t) => ({ sys: t.sys, name: t.name, value: t.s })), unit: "s", better: "lower" },
        { title: "large-v3-turbo, one 8-minute file, seconds, lower is better", rows: D.speech.turboLong.map((t) => ({ sys: t.sys, name: t.name, value: t.s })), unit: "s", better: "lower" },
        { title: "tiny, 73 clips, seconds, lower is better", rows: D.speech.tinyClips.map((t) => ({ sys: t.sys, name: t.name, value: t.s })), unit: "s", better: "lower" },
      ],
      note: `<p><strong>Level with MLX, ahead of everything else</strong>: 1.2 to 2.3x whisper.cpp, 1.9 to 6.7x PyTorch, 15 to 21x Sapient. Ahead of MLX on long audio, 2 to 5% behind on short clips.</p>
             <p>The encoder runs on Core ML (the GPU: on the Neural Engine it was slower and less accurate), the decoder on the engine's own Metal kernels. Word error rate equals PyTorch's: 3.25% for turbo.</p>`,
    },
    vit: {
      title: "A 631M-parameter ViT",
      sub: "one forward pass, GPU only, 1280 wide and 32 blocks",
      charts: [{ title: "milliseconds, lower is better", rows: D.vit.map((t) => ({ sys: t.sys, name: t.name, value: t.ms })), unit: "ms", better: "lower" }],
      note: `<p><strong>The engine's own graph and kernels</strong>, lowered from its IR to ggml's Metal backend: level with MLX and 3% ahead of PyTorch's fp16.</p>
             <p>Two profile-driven changes got it here: fused attention (153 to 107 ms) and f16 weight matrices (107 to 101 ms).</p>`,
    },
  };

  let current = "vla";
  function renderTab(key, animate) {
    current = key;
    const t = TABS[key];
    $("#bench-title").textContent = t.title;
    $("#bench-sub").textContent = t.sub;
    const host = $("#bench-charts");
    host.innerHTML = "";
    const systems = new Set();
    t.charts.forEach((c) => {
      const box = document.createElement("div");
      box.className = "sub-chart";
      if (t.charts.length > 1) box.innerHTML = `<div class="sub-title">${c.title}</div>`;
      else $("#bench-sub").textContent = `${t.sub}; ${c.title}`;
      const chart = document.createElement("div");
      chart.className = "chart";
      box.append(chart);
      host.append(box);
      const ref = c.rows.find((r) => r.sys === "dooms");
      barChart(chart, c.rows, c.unit, c.better, { ref: ref && ref.value, immediate: animate });
      c.rows.forEach((r) => systems.add(r.sys));
    });
    legend($("#bench-legend"), Object.keys(D.systems).filter((s) => systems.has(s)));
    $("#bench-note").innerHTML = typeof t.note === "function" ? t.note() : t.note;
    // table view: the same numbers, readable without color
    $("#bench-table").innerHTML = t.charts.map((c) => `
      <table><caption class="sub-title" style="text-align:left">${c.title}</caption>
      <thead><tr><th>configuration</th><th class="num">${c.unit}</th></tr></thead>
      <tbody>${c.rows.map((r) => `<tr><td>${r.name}</td><td class="num">${fmt(r.value)}</td></tr>`).join("")}</tbody></table>`).join("");
  }
  $$(".tabs button").forEach((b) => b.addEventListener("click", () => {
    $$(".tabs button").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
    renderTab(b.dataset.tab, true);
  }));
  renderTab("vla", false);
  $("#table-toggle").addEventListener("click", (e) => {
    const t = $("#bench-table");
    t.hidden = !t.hidden;
    e.currentTarget.setAttribute("aria-expanded", String(!t.hidden));
    e.currentTarget.innerHTML = t.hidden ? '<i class="ph ph-table"></i> Show as table' : '<i class="ph ph-chart-bar-horizontal"></i> Show as chart';
    $("#bench-charts").hidden = !t.hidden;
  });

  // ---------------------------------------------------------------- engineering log: numbers count from before to after when the entry arrives
  (function log() {
    const list = $("#log-list");
    D.log.forEach((e) => {
      const li = document.createElement("li");
      li.className = "entry";
      const dec = Number.isInteger(e.after) && Number.isInteger(e.before) ? 0 : e.after < 10 ? 2 : 1;
      li.innerHTML = `
        <div><h3>${e.title}</h3><p>${e.body}</p></div>
        <div class="delta">
          <div class="what">${e.what}</div>
          <div class="nums">${e.before.toFixed(dec)} <span aria-hidden="true">to</span></div>
          <span class="after${e.worse ? " worse" : ""}" data-from="${e.before}" data-to="${e.after}" data-dec="${dec}">${e.after.toFixed(dec)}</span><span class="unit">${e.unit}</span>
        </div>`;
      list.append(li);
      onReveal(li, () => {
        li.classList.add("in");
        const n = $(".after", li);
        if (reduce) return;
        const from = +n.dataset.from, to = +n.dataset.to, d = +n.dataset.dec, t0 = performance.now(), dur = 900;
        const step = (now) => {
          const k = Math.min(1, (now - t0) / dur), ease = 1 - Math.pow(1 - k, 3);
          n.textContent = (from + (to - from) * ease).toFixed(d);
          if (k < 1) requestAnimationFrame(step);
        };
        requestAnimationFrame(step);
      });
    });
  })();

  // ---------------------------------------------------------------- install tabs and copy
  $$(".code-tabs button").forEach((b) => b.addEventListener("click", () => {
    $$(".code-tabs button").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
    $$(".code-card pre").forEach((p) => (p.hidden = p.dataset.code !== b.dataset.code));
  }));
  $(".copy").addEventListener("click", async (e) => {
    const pre = $$(".code-card pre").find((p) => !p.hidden);
    const btn = e.currentTarget;
    try {
      await navigator.clipboard.writeText(pre.innerText);
      btn.innerHTML = '<i class="ph ph-check"></i>';
    } catch (err) {
      btn.innerHTML = '<i class="ph ph-x"></i>';
    }
    setTimeout(() => (btn.innerHTML = '<i class="ph ph-copy"></i>'), 1400);
  });

  // ---------------------------------------------------------------- theme toggle (system by default, remembered when chosen)
  $("#theme").addEventListener("click", () => {
    const root = document.documentElement;
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : !window.matchMedia("(prefers-color-scheme: light)").matches;
    const apply = () => { root.dataset.theme = dark ? "light" : "dark"; try { localStorage.setItem("dooms.theme", root.dataset.theme); } catch (err) {} };
    if (!reduce && document.startViewTransition) document.startViewTransition(apply); else apply();
  });


  // ---------------------------------------------------------------- hero number: from PyTorch's time down to the engine's
  (function heroCount() {
    const n = $("#hero-count");
    const to = Math.round(D.vla.totals[0].ms), from = Math.round(D.vla.totals.find((t) => t.name === "PyTorch MPS fp32").ms);
    if (reduce) { n.textContent = to; return; }
    n.textContent = from;
    const t0 = performance.now() + 450, dur = 1500;
    const step = (now) => {
      const k = Math.max(0, Math.min(1, (now - t0) / dur)), ease = 1 - Math.pow(1 - k, 4);
      n.textContent = Math.round(from + (to - from) * ease);
      if (k < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  })();

  // ---------------------------------------------------------------- section headings and loss figures rise as they arrive
  $$(".rise-in").forEach((el) => onReveal(el, () => {
    el.classList.add("in");
    $$("[data-count]", el).forEach((c) => {
      if (reduce) return;
      const to = +c.dataset.count, suffix = c.dataset.suffix || "", d = to % 1 ? 2 : 0, t0 = performance.now(), dur = 1100;
      const step = (now) => {
        const k = Math.min(1, (now - t0) / dur), ease = 1 - Math.pow(1 - k, 3);
        c.textContent = (to * ease).toFixed(d) + suffix;
        if (k < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    });
  }));
  if (!io) $$(".rise-in").forEach((el) => el.classList.add("in"));

  // ---------------------------------------------------------------- GitHub: live repository and profile details (the static text stays if the API is unreachable)
  fetch("https://api.github.com/repos/jay7828/dooms").then((r) => (r.ok ? r.json() : null)).then((d) => {
    if (!d) return;
    $("#repo-stars").textContent = d.stargazers_count;
    $("#repo-forks").textContent = d.forks_count;
    if (d.description) $("#repo-desc").textContent = d.description;
  }).catch(() => {});
  fetch("https://api.github.com/users/jay7828").then((r) => (r.ok ? r.json() : null)).then((d) => {
    if (!d) return;
    if (d.name) $("#author-name").textContent = d.name;
    if (d.avatar_url) $("#author-avatar").src = d.avatar_url + "&s=160";
    $("#author-meta").textContent = `${d.public_repos} public repositories`;
  }).catch(() => {});

  $("#machine").textContent = D.machine;
  $("#measured").textContent = D.measured;
})();
