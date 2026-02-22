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
      const result = await Promise.race([
        executeInTab(async (p, name, currentPlatform) => {
        const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

        const setDraftEditorValue = (el, value) => {
          const text = String(value || "");
          const root = el.closest(".DraftEditor-root") || el;
          const editable = root.querySelector(".public-DraftEditor-content[contenteditable='true']") || el;
          const textSpan = editable.querySelector("span[data-text='true']");

          editable.focus();
          editable.dispatchEvent(new MouseEvent("mousedown", { bubbles: true, cancelable: true, composed: true }));
          editable.dispatchEvent(new MouseEvent("mouseup", { bubbles: true, cancelable: true, composed: true }));
          editable.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true, composed: true }));

          let applied = false;
          try {
            // Simulated paste path for DraftJS-like editors.
            const data = new DataTransfer();
            data.setData("text/plain", text);
            const pasteEvt = new ClipboardEvent("paste", { bubbles: true, cancelable: true, clipboardData: data });
            editable.dispatchEvent(pasteEvt);
            applied = String(editable.textContent || "").trim().length > 0;
          } catch (_) {}

          if (!applied) {
            try { document.execCommand("selectAll", false, null); } catch (_) {}
            try {
              applied = Boolean(document.execCommand("insertText", false, text));
            } catch (_) {
              applied = false;
            }
          }

          if (!applied || String(editable.textContent || "").trim().length === 0) {
            if (textSpan) {
              textSpan.textContent = text;
            } else {
              editable.textContent = text;
            }
          }

          try { editable.dispatchEvent(new InputEvent("beforeinput", { bubbles: true, composed: true, data: text, inputType: "insertFromPaste" })); } catch (_) {}
          try { editable.dispatchEvent(new InputEvent("input", { bubbles: true, composed: true, data: text, inputType: "insertFromPaste" })); } catch (_) {
            editable.dispatchEvent(new Event("input", { bubbles: true }));
          }
          editable.dispatchEvent(new Event("change", { bubbles: true }));
          editable.dispatchEvent(new Event("blur", { bubbles: true }));
          return true;
        };

        const setValue = (el, value) => {
          el.focus();
          const isDraftEditor = (
            String(el.getAttribute("role") || "").toLowerCase() === "combobox"
            || String(el.className || "").includes("DraftEditor")
            || Boolean(el.closest(".DraftEditor-root"))
          );
          if (el.isContentEditable || String(el.getAttribute("contenteditable") || "").toLowerCase() === "true") {
            if (isDraftEditor) {
              return setDraftEditorValue(el, value);
            }
            el.textContent = value;
            el.dispatchEvent(new InputEvent("beforeinput", { bubbles: true, composed: true, data: value, inputType: "insertText" }));
            el.dispatchEvent(new InputEvent("input", { bubbles: true, composed: true, data: value, inputType: "insertText" }));
            el.dispatchEvent(new Event("change", { bubbles: true }));
            return true;
          }
          const proto = Object.getPrototypeOf(el);
          const setter = Object.getOwnPropertyDescriptor(proto, "value")?.set;
          if (setter) setter.call(el, value); else el.value = value;
          el.dispatchEvent(new Event("input", { bubbles: true }));
          el.dispatchEvent(new Event("change", { bubbles: true }));
          return true;
        };

        const typeToken = async (el, token) => {
          if (!el) return;
          el.focus();
          if (el.isContentEditable) {
            if (document.execCommand) {
              document.execCommand("insertText", false, token);
            } else {
              el.textContent = `${el.textContent || ""}${token}`;
              el.dispatchEvent(new InputEvent("input", { bubbles: true, data: token, inputType: "insertText" }));
            }
          } else {
            el.value = `${el.value || ""}${token}`;
            el.dispatchEvent(new Event("input", { bubbles: true }));
          }
          await wait(120);
        };

        const clickHashtagSuggestion = async () => {
          const selectors = [
            "[data-e2e*='hashtag'] li",
            "[data-e2e*='hashtag'] button",
            "[role='listbox'] [role='option']",
            "div[class*='mention'] [role='option']",
            "div[class*='hashtag'] [role='option']"
          ];
          for (const sel of selectors) {
            const candidate = document.querySelector(sel);
            if (candidate) {
              candidate.scrollIntoView({ block: "center", inline: "center" });
              candidate.dispatchEvent(new MouseEvent("mousedown", { bubbles: true, cancelable: true, composed: true }));
              candidate.dispatchEvent(new MouseEvent("mouseup", { bubbles: true, cancelable: true, composed: true }));
              candidate.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true, composed: true }));
              await wait(120);
              return true;
            }
          }
          return false;
        };

        if (name === "dom.type") {
          const el = document.querySelector(p.selector || "");
          if (!el) return { typed: false, selector: p.selector || "" };
          return { typed: setValue(el, p.value ?? p.text ?? ""), selector: p.selector || "" };
        }

        const fields = p.fields || {};
        const selectors = {
          title: ["textarea#title-textarea", "#textbox", "input[name='title']", "textarea[name='title']"],
          description: [
            ".DraftEditor-editorContainer [contenteditable='true'][role='combobox']",
            "div.public-DraftEditor-content[contenteditable='true'][role='combobox']",
            "div[contenteditable='true'][role='combobox']",
            "div[contenteditable='true'][role='textbox']",
            "div[contenteditable='true']",
            "textarea#description-textarea",
            "textarea[name='description']"
          ]
        };
        const out = {};
        for (const [key, value] of Object.entries(fields)) {
          const list = selectors[key] || [`[name='${key}']`, `textarea[name='${key}']`, `input[name='${key}']`];
          const el = list.map((sel) => document.querySelector(sel)).find(Boolean);
          if (!el) {
            out[key] = false;
            continue;
          }

          const text = String(value || "");
          if (currentPlatform === "tiktok" && key === "description") {
            const tiktokSelectors = [
              ".DraftEditor-editorContainer [contenteditable='true'][role='combobox']",
              "div.public-DraftEditor-content[contenteditable='true'][role='combobox']",
              "div[contenteditable='true'][role='combobox']",
              "div[contenteditable='true']",
            ];
            const tiktokEl = tiktokSelectors.map((sel) => document.querySelector(sel)).find(Boolean) || el;
            out[key] = setValue(tiktokEl, text);
          } else {
            out[key] = setValue(el, text);
          }
        }
        return out;
        }, [payload, msg.name, platform], platform),
        new Promise((resolve) => setTimeout(() => resolve({ timeout: true, typed: false }), 25000)),
      ]);
      if (result && result.timeout) {
        ackCmd(msg, false, {}, "form.fill timed out in extension script");
      } else {
        ackCmd(msg, true, result);
      }
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
