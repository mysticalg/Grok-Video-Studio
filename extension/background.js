const WS_URL = "ws://127.0.0.1:18792";
const SCHEMA_VERSION = 1;
const CLIENT_ID = "relay-extension";

let socket = null;
let reconnectDelayMs = 1000;
const platformState = {
  youtube: { state: "ready" },
  tiktok: { state: "ready" },
  facebook: { state: "ready" }
};

const POST_SELECTORS = {
  tiktok: [
    "button[data-e2e=\"post_video_button\"]",
    "button[aria-label*=\"Post\"]"
  ],
  youtube: ["#done-button", "button[aria-label*=\"Publish\"]"],
  facebook: ["div[aria-label*=\"Publish\"]", "button[aria-label*=\"Publish\"]"]
};

const DRAFT_SELECTORS = {
  tiktok: [
    "button[data-e2e=\"save_draft_button\"]",
    "button[aria-label*=\"Draft\"]",
    "button[class*=\"draft\"]"
  ]
};

function makeId() {
  return `msg_${crypto.randomUUID().replace(/-/g, "")}`;
}

function sendMessage(msg) {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(msg));
  }
}

function sendEvent(name, payload) {
  sendMessage({ v: SCHEMA_VERSION, type: "event", id: makeId(), name, payload: payload || {} });
}

function ackCmd(msg, ok, payload, error) {
  const response = { v: SCHEMA_VERSION, type: "cmd_ack", id: msg.id, name: msg.name, ok, payload: payload || {} };
  if (!ok && error) response.error = String(error);
  sendMessage(response);
}

async function executeInTab(fn, args = [], platform = "") {
  const normalized = String(platform || "").toLowerCase();
  const matchByPlatform = {
    tiktok: ["tiktok.com/tiktokstudio/upload", "tiktok.com/upload", "tiktok.com"],
    youtube: ["studio.youtube.com", "youtube.com"],
    facebook: ["facebook.com/reels/create", "facebook.com"]
  };

  const allTabs = await chrome.tabs.query({});
  const targets = matchByPlatform[normalized] || [];
  const platformTab = allTabs.find((tab) => tab.id != null && targets.some((host) => String(tab.url || "").includes(host)));

  let activeTab = allTabs.find((tab) => tab.active && tab.lastFocusedWindow && tab.id != null);
  if (!activeTab) {
    activeTab = allTabs.find((tab) => tab.active && tab.id != null);
  }

  const chosen = platformTab || activeTab;
  if (!chosen || chosen.id == null) {
    throw new Error(`No suitable tab available for platform=${normalized || "unknown"}`);
  }

  const results = await chrome.scripting.executeScript({ target: { tabId: chosen.id }, func: fn, args });
  return results[0]?.result;
}

async function handleCmd(msg) {
  try {
    const payload = msg.payload || {};

    if (msg.name === "dom.ping") {
      const result = await executeInTab(() => ({ title: document.title, href: window.location.href }), [], payload.platform || "");
      ackCmd(msg, true, result);
      return;
    }

    if (msg.name === "platform.ensure_logged_in") {
      ackCmd(msg, true, { platform: payload.platform || "unknown", loggedIn: true });
      return;
    }

    if (msg.name === "dom.query") {
      const result = await executeInTab((selector) => ({ count: document.querySelectorAll(selector).length }), [payload.selector || "body"], payload.platform || "");
      ackCmd(msg, true, result);
      return;
    }

    if (msg.name === "dom.click" || msg.name === "post.submit") {
      const platform = String(payload.platform || "").toLowerCase();
      const isDraft = String(payload.mode || payload.publishMode || "").toLowerCase() === "draft";
      const submitSelectors = isDraft ? (DRAFT_SELECTORS[platform] || []) : (POST_SELECTORS[platform] || []);
      const candidates = msg.name === "post.submit"
        ? (submitSelectors.length ? submitSelectors : [payload.selector || "button"])
        : [payload.selector || "button"];
      const result = await executeInTab(async (selectors, opts) => {
        const isVisible = (el) => !!(el && (el.offsetWidth || el.offsetHeight || el.getClientRects().length));
        const isEnabled = (el) => {
          if (!el) return false;
          const attrDisabled = String(el.getAttribute("aria-disabled") || "").toLowerCase() === "true";
          const dataDisabled = String(el.getAttribute("data-disabled") || "").toLowerCase() === "true";
          const htmlDisabled = Boolean(el.disabled);
          const loading = String(el.getAttribute("data-loading") || "").toLowerCase() === "true";
          return !(attrDisabled || dataDisabled || htmlDisabled || loading);
        };

        const hasUploadReadySignal = () => {
          const status = document.querySelector("div.info-status.success");
          if (!status) return false;
          const text = (status.textContent || "").toLowerCase();
          return text.includes("uploaded");
        };

        const findCandidate = () => {
          for (const sel of selectors) {
            const el = Array.from(document.querySelectorAll(sel)).find(isVisible) || document.querySelector(sel);
            if (el) return { el, selector: sel };
          }
          return null;
        };

        const timeoutMs = Number(opts.timeoutMs || 30000);
        const started = Date.now();
        while (Date.now() - started < timeoutMs) {
          const found = findCandidate();
          if (found && isEnabled(found.el) && (!opts.waitForUpload || hasUploadReadySignal())) {
            found.el.scrollIntoView({ block: "center", inline: "center" });
            ["pointerdown", "mousedown", "pointerup", "mouseup", "click"].forEach((evt) => {
              found.el.dispatchEvent(new MouseEvent(evt, { bubbles: true, cancelable: true, composed: true }));
            });
            return { clicked: true, selector: found.selector, waitedMs: Date.now() - started };
          }
          await new Promise((resolve) => setTimeout(resolve, 250));
        }

        const found = findCandidate();
        if (!found) return { clicked: false, reason: "not_found", selectors };
        if (!isEnabled(found.el)) return { clicked: false, reason: "disabled", selector: found.selector };
        if (opts.waitForUpload && !hasUploadReadySignal()) return { clicked: false, reason: "upload_not_ready", selector: found.selector };
        return { clicked: false, reason: "unknown", selector: found.selector };
      }, [candidates, { waitForUpload: Boolean(payload.waitForUpload), timeoutMs: Number(payload.timeoutMs || 30000) }], platform);
      if (msg.name === "post.submit") {
        if (platformState[platform]) {
          if (result?.clicked) {
            platformState[platform].state = isDraft ? "drafted" : "submitted";
          } else {
            platformState[platform].state = isDraft ? "draft_button_not_ready" : "submit_not_found";
          }
        }
        sendEvent("state", {
          platform,
          state: result?.clicked
            ? (isDraft ? "draft_clicked" : "post_clicked")
            : (isDraft ? "draft_button_not_ready" : "post_button_not_found"),
          selector: result?.selector,
          reason: result?.reason,
        });
      }
      ackCmd(msg, true, result);
      return;
    }

    if (msg.name === "dom.type" || msg.name === "form.fill") {
      const platform = String(payload.platform || "").toLowerCase();
      const result = await executeInTab(async (p, name, currentPlatform) => {
        const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
        const isVisible = (el) => !!(el && (el.offsetWidth || el.offsetHeight || el.getClientRects().length));
        const normalize = (v) => String(v || "").replace(/\u200B/g, "").replace(/\s+/g, " ").trim();

        const activate = (el) => {
          if (!el) return;
          try { el.scrollIntoView({ block: "center", inline: "center" }); } catch (_) {}
          try {
            ["pointerdown", "mousedown", "pointerup", "mouseup", "click"].forEach((evt) => {
              el.dispatchEvent(new MouseEvent(evt, { bubbles: true, cancelable: true, composed: true }));
            });
            el.click?.();
          } catch (_) {}
          try { el.focus(); } catch (_) {}
        };

        const setReactLikeValue = (el, value) => {
          const next = String(value || "");
          if (!el) return { ok: false, method: "missing", len: 0 };
          activate(el);
          const isEditable = el.isContentEditable || String(el.getAttribute("contenteditable") || "").toLowerCase() === "true";

          if (isEditable) {
            let method = "contenteditable_execCommand";
            try {
              try {
                const sel = window.getSelection();
                const range = document.createRange();
                range.selectNodeContents(el);
                if (sel) {
                  sel.removeAllRanges();
                  sel.addRange(range);
                }
              } catch (_) {}
              try { document.execCommand?.("selectAll", false, null); } catch (_) {}
              try { document.execCommand?.("insertText", false, next); } catch (_) {}
              if (normalize(el.textContent) !== normalize(next)) {
                method = "contenteditable_textContent";
                el.textContent = "";
                el.textContent = next;
              }
              try { el.dispatchEvent(new InputEvent("beforeinput", { bubbles: true, composed: true, data: next, inputType: "insertText" })); } catch (_) {}
              try { el.dispatchEvent(new InputEvent("input", { bubbles: true, composed: true, data: next, inputType: "insertText" })); } catch (_) {
                el.dispatchEvent(new Event("input", { bubbles: true }));
              }
              el.dispatchEvent(new Event("change", { bubbles: true }));
              el.dispatchEvent(new Event("blur", { bubbles: true }));
            } catch (_) {
              return { ok: false, method: "contenteditable_failed", len: normalize(el.textContent).length };
            }
            return { ok: normalize(el.textContent) === normalize(next), method, len: normalize(el.textContent).length };
          }

          try {
            const proto = Object.getPrototypeOf(el);
            const setter = Object.getOwnPropertyDescriptor(proto, "value")?.set;
            if (setter) setter.call(el, next); else el.value = next;
            el.dispatchEvent(new InputEvent("input", { bubbles: true, composed: true, data: next, inputType: "insertText" }));
            el.dispatchEvent(new Event("change", { bubbles: true }));
            return { ok: normalize(el.value) === normalize(next), method: "value_setter", len: normalize(el.value).length };
          } catch (_) {
            return { ok: false, method: "value_failed", len: 0 };
          }
        };

        const waitForSelector = async (selectors, timeoutMs = 10000) => {
          const started = Date.now();
          while (Date.now() - started < timeoutMs) {
            for (const sel of selectors) {
              try {
                const matches = Array.from(document.querySelectorAll(sel));
                const visible = matches.find(isVisible) || matches[0];
                if (visible) return { el: visible, selector: sel };
              } catch (_) {}
            }
            await wait(180);
          }
          return { el: null, selector: "" };
        };

        const findYoutubeByLabel = (labelHint) => {
          const containers = Array.from(document.querySelectorAll("ytcp-form-input-container"));
          for (const container of containers) {
            const label = String(container.querySelector("#label-text")?.textContent || "").toLowerCase();
            if (!label.includes(labelHint)) continue;
            const outer = container.querySelector("#outer, #child-input, #container-content");
            if (outer) activate(outer);
            const tb = container.querySelector('#textbox[contenteditable="true"], textarea#textbox');
            if (tb) return tb;
          }
          return null;
        };

        const findBestTextbox = () => {
          const candidates = Array.from(document.querySelectorAll("textarea,[contenteditable='true'],[role='textbox']"))
            .filter((el) => isVisible(el) && !el.disabled);
          candidates.sort((a, b) => {
            const ra = a.getBoundingClientRect();
            const rb = b.getBoundingClientRect();
            return (rb.width * rb.height) - (ra.width * ra.height);
          });
          return candidates[0] || null;
        };

        if (name === "dom.type") {
          const typedValue = String(p.value ?? p.text ?? "");
          const selector = String(p.selector || "");
          const timeoutMs = Number(p.timeoutMs || 10000);
          const { el, selector: used } = await waitForSelector([selector], timeoutMs);
          if (!el) return { typed: false, selector, reason: "not_found" };
          const wrote = setReactLikeValue(el, typedValue);
          return { typed: Boolean(wrote.ok), selector: used || selector, method: wrote.method, len: wrote.len };
        }

        const fields = p.fields || {};
        const timeoutMs = Number(p.timeoutMs || 12000);
        const selectorsByField = {
          title: [
            '#title-textarea #textbox[contenteditable="true"]',
            'div#textbox[contenteditable="true"][aria-label*="title" i]',
            'textarea#textbox[aria-label*="title" i]',
            "input[name='title']",
            "textarea[name='title']",
          ],
          description: [
            '#description #textbox[contenteditable="true"]',
            'div#textbox[contenteditable="true"][aria-label*="tell viewers" i]',
            'div#textbox[contenteditable="true"][aria-label*="description" i]',
            '[contenteditable="true"][aria-placeholder*="describe your reel" i]',
            '[contenteditable="true"][data-lexical-editor="true"]',
            "textarea[name='description']",
            "div[contenteditable='true'][role='textbox']",
            "div[contenteditable='true']",
          ],
        };

        const out = {};
        const diagnostics = [];
        for (const [key, value] of Object.entries(fields)) {
          const text = String(value || "");
          let el = null;
          let usedSelector = "";

          if (currentPlatform === "youtube") {
            el = key === "title" ? findYoutubeByLabel("title") : (key === "description" ? findYoutubeByLabel("description") : null);
            if (el) usedSelector = `ytcp-form-input-container[label~=${key}]`;
          }

          if (!el) {
            const list = selectorsByField[key] || [`[name='${key}']`, `textarea[name='${key}']`, `input[name='${key}']`];
            const found = await waitForSelector(list, timeoutMs);
            el = found.el;
            usedSelector = found.selector;
          }

          if (!el && (key === "description" || key === "title")) {
            const fallback = findBestTextbox();
            if (fallback) {
              el = fallback;
              usedSelector = "fallback_best_textbox";
            }
          }

          if (!el) {
            out[key] = false;
            diagnostics.push({ field: key, method: "not_found", selector: usedSelector, len: 0 });
            continue;
          }

          const wrote = setReactLikeValue(el, text);
          out[key] = Boolean(wrote.ok);
          diagnostics.push({
            field: key,
            method: wrote.method,
            selector: usedSelector,
            tag: String(el.tagName || ""),
            role: String(el.getAttribute?.("role") || ""),
            contenteditable: String(el.getAttribute?.("contenteditable") || ""),
            len: Number(wrote.len || 0),
          });
        }

        return { ...out, diagnostics };
      }, [payload, msg.name, platform], platform);
      ackCmd(msg, true, result);
      return;
    }


    if (msg.name === "upload.select_file") {
      sendEvent("error", {
        message: "Extension cannot set file path directly without CDP file-input support",
        diagnostic: { platform: payload.platform || "", selector: "input[type=file]" }
      });
      ackCmd(msg, true, { requiresUserAction: true });
      return;
    }

    if (msg.name === "post.status") {
      const platform = String(payload.platform || "").toLowerCase();
      ackCmd(msg, true, platformState[platform] || { state: "ready" });
      return;
    }

    ackCmd(msg, false, {}, `Unsupported cmd: ${msg.name}`);
  } catch (err) {
    sendEvent("error", { message: err?.message || String(err) });
    ackCmd(msg, false, {}, err?.message || String(err));
  }
}

function connect() {
  socket = new WebSocket(WS_URL);

  socket.onopen = () => {
    reconnectDelayMs = 1000;
    sendMessage({ v: SCHEMA_VERSION, type: "hello", id: makeId(), name: "hello", payload: { client_id: CLIENT_ID } });
    sendEvent("log", { message: "relay extension connected" });
  };

  socket.onmessage = async (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    if (!msg || !msg.type || !msg.id) return;
    if (msg.type === "heartbeat") {
      sendMessage({ v: SCHEMA_VERSION, type: "heartbeat_ack", id: msg.id, name: "heartbeat_ack", payload: {} });
      return;
    }
    if (msg.type === "cmd") await handleCmd(msg);
  };

  socket.onclose = () => {
    setTimeout(connect, reconnectDelayMs);
    reconnectDelayMs = Math.min(reconnectDelayMs * 2, 30000);
  };
  socket.onerror = () => socket?.close();
}

connect();
