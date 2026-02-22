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

async function executeInActiveTab(fn, args = []) {
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tabs.length || tabs[0].id == null) throw new Error("No active tab available");
  const tabId = tabs[0].id;
  const results = await chrome.scripting.executeScript({ target: { tabId }, func: fn, args });
  return results[0]?.result;
}

async function handleCmd(msg) {
  try {
    const payload = msg.payload || {};

    if (msg.name === "dom.ping") {
      const result = await executeInActiveTab(() => ({ title: document.title, href: window.location.href }));
      ackCmd(msg, true, result);
      return;
    }

    if (msg.name === "platform.ensure_logged_in") {
      ackCmd(msg, true, { platform: payload.platform || "unknown", loggedIn: true });
      return;
    }

    if (msg.name === "dom.query") {
      const result = await executeInActiveTab((selector) => ({ count: document.querySelectorAll(selector).length }), [payload.selector || "body"]);
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
      const result = await executeInActiveTab((selectors) => {
        const isVisible = (el) => !!(el && (el.offsetWidth || el.offsetHeight || el.getClientRects().length));
        for (const sel of selectors) {
          const el = Array.from(document.querySelectorAll(sel)).find(isVisible) || document.querySelector(sel);
          if (!el) continue;
          el.scrollIntoView({ block: "center", inline: "center" });
          ["pointerdown", "mousedown", "pointerup", "mouseup", "click"].forEach((evt) => {
            el.dispatchEvent(new MouseEvent(evt, { bubbles: true, cancelable: true, composed: true }));
          });
          return { clicked: true, selector: sel };
        }
        return { clicked: false, reason: "not_found", selectors };
      }, [candidates]);
      if (msg.name === "post.submit") {
        if (platformState[platform]) {
          if (result?.clicked) {
            platformState[platform].state = isDraft ? "drafted" : "submitted";
          } else {
            platformState[platform].state = isDraft ? "draft_button_not_found" : "submit_not_found";
          }
        }
        sendEvent("state", {
          platform,
          state: result?.clicked
            ? (isDraft ? "draft_clicked" : "post_clicked")
            : (isDraft ? "draft_button_not_found" : "post_button_not_found"),
          selector: result?.selector,
        });
      }
      ackCmd(msg, true, result);
      return;
    }

    if (msg.name === "dom.type" || msg.name === "form.fill") {
      const result = await executeInActiveTab((p, name) => {
        const setValue = (el, value) => {
          el.focus();
          if (el.isContentEditable) {
            el.textContent = value;
            el.dispatchEvent(new InputEvent("input", { bubbles: true }));
            return true;
          }
          const proto = Object.getPrototypeOf(el);
          const setter = Object.getOwnPropertyDescriptor(proto, "value")?.set;
          if (setter) setter.call(el, value); else el.value = value;
          el.dispatchEvent(new Event("input", { bubbles: true }));
          el.dispatchEvent(new Event("change", { bubbles: true }));
          return true;
        };

        if (name === "dom.type") {
          const el = document.querySelector(p.selector || "");
          if (!el) return { typed: false, selector: p.selector || "" };
          return { typed: setValue(el, p.text || ""), selector: p.selector || "" };
        }

        const fields = p.fields || {};
        const selectors = {
          title: ["textarea#title-textarea", "#textbox", "input[name='title']", "textarea[name='title']"],
          description: ["textarea#description-textarea", "div[contenteditable='true']", "textarea[name='description']"]
        };
        const out = {};
        for (const [key, value] of Object.entries(fields)) {
          const list = selectors[key] || [`[name='${key}']`, `textarea[name='${key}']`, `input[name='${key}']`];
          const el = list.map((s) => document.querySelector(s)).find(Boolean);
          out[key] = el ? setValue(el, String(value || "")) : false;
        }
        return out;
      }, [payload, msg.name]);
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
