const WS_URL = "ws://127.0.0.1:18792";
const SCHEMA_VERSION = 1;
const CLIENT_ID = "relay-extension";

let socket = null;
let reconnectDelayMs = 1000;

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
  const response = {
    v: SCHEMA_VERSION,
    type: "cmd_ack",
    id: msg.id,
    name: msg.name,
    ok,
    payload: payload || {}
  };
  if (!ok && error) {
    response.error = String(error);
  }
  sendMessage(response);
}

async function executeInActiveTab(fn, args = []) {
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tabs.length || tabs[0].id == null) {
    throw new Error("No active tab available");
  }
  const tabId = tabs[0].id;
  const results = await chrome.scripting.executeScript({
    target: { tabId },
    func: fn,
    args
  });
  return results[0]?.result;
}

async function handleCmd(msg) {
  try {
    if (msg.name === "dom.ping") {
      const result = await executeInActiveTab(() => ({
        title: document.title,
        href: window.location.href,
        bodyText: (document.body?.innerText || "").slice(0, 120)
      }));
      ackCmd(msg, true, result);
      return;
    }

    if (msg.name === "youtube.ui.check") {
      const selectors = msg.payload?.selectors || ["input[type='file']", "button", "ytd-app"];
      const result = await executeInActiveTab((s) => {
        const checks = {};
        for (const selector of s) {
          checks[selector] = !!document.querySelector(selector);
        }
        return checks;
      }, [selectors]);
      ackCmd(msg, true, result);
      return;
    }

    ackCmd(msg, false, {}, `Unsupported cmd: ${msg.name}`);
  } catch (err) {
    ackCmd(msg, false, {}, err?.message || String(err));
  }
}

function connect() {
  socket = new WebSocket(WS_URL);

  socket.onopen = () => {
    reconnectDelayMs = 1000;
    sendMessage({
      v: SCHEMA_VERSION,
      type: "hello",
      id: makeId(),
      name: "hello",
      payload: { client_id: CLIENT_ID, ua: navigator.userAgent }
    });
    sendEvent("extension.connected", { client_id: CLIENT_ID });
  };

  socket.onmessage = async (ev) => {
    let msg = null;
    try {
      msg = JSON.parse(ev.data);
    } catch {
      return;
    }

    if (!msg || !msg.type || !msg.id) return;

    if (msg.type === "heartbeat") {
      sendMessage({ v: SCHEMA_VERSION, type: "heartbeat_ack", id: msg.id, name: "heartbeat_ack", payload: {} });
      return;
    }

    if (msg.type === "cmd") {
      await handleCmd(msg);
    }
  };

  socket.onclose = () => {
    sendEvent("extension.disconnected", { retry_in_ms: reconnectDelayMs });
    setTimeout(connect, reconnectDelayMs);
    reconnectDelayMs = Math.min(reconnectDelayMs * 2, 30000);
  };

  socket.onerror = () => {
    socket?.close();
  };
}

connect();
