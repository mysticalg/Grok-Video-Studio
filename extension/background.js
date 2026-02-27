const WS_URL = "ws://127.0.0.1:18792";
const SCHEMA_VERSION = 1;
const CLIENT_ID = "relay-extension";

let socket = null;

async function findTargetTab(platform = "") {
  const normalized = String(platform || "").toLowerCase();
  const matchByPlatform = {
    tiktok: ["tiktok.com/tiktokstudio/upload", "tiktok.com/upload", "tiktok.com"],
    // Keep YouTube matching scoped to Studio so a regular watch tab doesn't
    // accidentally receive form.fill commands meant for the upload dialog.
    youtube: ["studio.youtube.com/channel", "studio.youtube.com"],
    facebook: ["facebook.com/reels/create", "facebook.com"],
    instagram: ["instagram.com/", "instagram.com"],
    x: ["x.com/compose/post", "x.com/compose", "x.com"]
  };

  const allTabs = await chrome.tabs.query({});
  const targets = matchByPlatform[normalized] || [];
  const platformTab = allTabs.find((tab) => tab.id != null && targets.some((host) => String(tab.url || "").includes(host)));

  let activeTab = allTabs.find((tab) => tab.active && tab.lastFocusedWindow && tab.id != null);
  if (!activeTab) activeTab = allTabs.find((tab) => tab.active && tab.id != null);

  const chosen = platformTab || activeTab;
  if (!chosen || chosen.id == null) {
    throw new Error(`No suitable tab available for platform=${normalized || "unknown"}`);
  }
  return chosen;
}

let reconnectDelayMs = 1000;
const platformState = {
  youtube: { state: "ready" },
  tiktok: { state: "ready" },
  facebook: { state: "ready" },
  instagram: { state: "ready" },
  x: { state: "ready" }
};

const POST_SELECTORS = {
  tiktok: [
    'button[data-e2e="post_video_button"]',
    'button[aria-label*="Post"]'
  ],
  youtube: ['#done-button', 'button[aria-label*="Publish"]'],
  facebook: ['div[aria-label*="Publish"]', 'button[aria-label*="Publish"]'],
  instagram: ['div[role="button"]:has-text("Share")', 'button:has-text("Share")', 'button[aria-label*="Share" i]', 'button[type="submit"]'],
  x: ['button[data-testid="tweetButton"]', 'button[data-testid="tweetButtonInline"]', 'div[data-testid="tweetButtonInline"]']
};

const DRAFT_SELECTORS = {
  tiktok: [
    "button[data-e2e=\"save_draft_button\"] div.Button__content",
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
  const chosen = await findTargetTab(platform);
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

    if (msg.name === "dom.click" || msg.name === "dom.click_random" || msg.name === "post.submit") {
      const platform = String(payload.platform || "").toLowerCase();
      const isDraft = String(payload.mode || payload.publishMode || "").toLowerCase() === "draft";
      const submitSelectors = isDraft ? (DRAFT_SELECTORS[platform] || []) : (POST_SELECTORS[platform] || []);
      const candidates = msg.name === "post.submit"
        ? (submitSelectors.length ? submitSelectors : [payload.selector || "button"])
        : [payload.selector || "button"];
      const useRandomMatch = msg.name === "dom.click_random";
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

        const textMatches = (el, needle) => {
          const expected = String(needle || "").trim().toLowerCase();
          if (!expected) return true;
          const text = String((el?.closest("button, [role='button']") || el)?.textContent || "").trim().toLowerCase();
          return text.includes(expected);
        };

        const hasUploadReadySignal = () => {
          const status = document.querySelector("div.info-status.success");
          if (status) {
            const text = (status.textContent || "").toLowerCase();
            if (text.includes("uploaded") || text.includes("complete")) return true;
          }
          try {
            const entries = [];
            entries.push(...performance.getEntriesByType("measure"));
            entries.push(...performance.getEntriesByType("mark"));
            entries.push(...performance.getEntriesByType("resource"));
            const hasPerfReady = entries.some((entry) => {
              const name = String(entry?.name || "").toLowerCase();
              return name.includes("video_ready") || name.includes("video_post_ready") || name.includes("upload") || name.includes("complete");
            });
            if (hasPerfReady) return true;
          } catch (_) {}
          return false;
        };

        const tryXFormSubmitFallback = () => {
          if (String(opts.platform || "").toLowerCase() !== "x" || opts.isDraft) return { clicked: false };
          const composer = document.querySelector("div[data-testid='tweetTextarea_0'][contenteditable='true']")
            || document.querySelector("div[data-testid^='tweetTextarea'][contenteditable='true']")
            || document.querySelector("div[role='textbox'][contenteditable='true'][aria-label*='post text' i]");
          const form = composer ? (composer.closest("form") || composer.parentElement?.closest("form")) : document.querySelector("form");

          const tweetButton = (form && (form.querySelector("button[data-testid='tweetButton']") || form.querySelector("button[data-testid='tweetButtonInline']")))
            || document.querySelector("button[data-testid='tweetButton']")
            || document.querySelector("button[data-testid='tweetButtonInline']");
          if (tweetButton && isEnabled(tweetButton)) {
            try { tweetButton.scrollIntoView({ block: "center", inline: "center" }); } catch (_) {}
            ["pointerdown", "mousedown", "pointerup", "mouseup", "click"].forEach((evt) => {
              try { tweetButton.dispatchEvent(new MouseEvent(evt, { bubbles: true, cancelable: true, composed: true })); } catch (_) {}
            });
            try { tweetButton.click?.(); } catch (_) {}
            return { clicked: true, selector: "button[data-testid='tweetButton']", mode: "x_button_fallback" };
          }

          if (form) {
            try { form.requestSubmit?.(); return { clicked: true, selector: "form", mode: "x_request_submit" }; } catch (_) {}
            try { form.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })); return { clicked: true, selector: "form", mode: "x_submit_event" }; } catch (_) {}
          }
          return { clicked: false };
        };

        const findCandidate = () => {
          for (const sel of selectors) {
            const visibleMatches = Array.from(document.querySelectorAll(sel)).filter((node) => isVisible(node) && textMatches(node, opts.textContains));
            if (visibleMatches.length > 0) {
              if (opts.useRandomMatch && visibleMatches.length > 1) {
                const idx = Math.floor(Math.random() * visibleMatches.length);
                return { el: visibleMatches[idx], selector: sel, randomIndex: idx, randomPoolSize: visibleMatches.length };
              }
              return { el: visibleMatches[0], selector: sel, randomIndex: 0, randomPoolSize: visibleMatches.length };
            }
            const fallback = document.querySelector(sel);
            if (fallback && textMatches(fallback, opts.textContains)) return { el: fallback, selector: sel, randomIndex: 0, randomPoolSize: 1 };
          }
          return null;
        };

        const timeoutMs = Math.max(1000, Number(opts.timeoutMs || 30000));
        const started = Date.now();
        while (Date.now() - started < timeoutMs) {
          const found = findCandidate();
          if (found && isEnabled(found.el) && (!opts.waitForUpload || hasUploadReadySignal())) {
            const clickTarget = found.el.closest("button, [role='button']") || found.el;
            const clickText = String(clickTarget.textContent || "").trim().toLowerCase();
            const clickE2E = String(clickTarget.getAttribute("data-e2e") || "").trim().toLowerCase();
            if (opts.isDraft && (clickE2E === "discard_post_button" || clickText.includes("discard"))) {
              return { clicked: false, reason: "blacklisted_discard", selector: found.selector };
            }
            clickTarget.scrollIntoView({ block: "center", inline: "center" });
            if (opts.singleClick) {
              try { clickTarget.click?.(); } catch (_) {
                clickTarget.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true, composed: true }));
              }
            } else {
              ["pointerdown", "mousedown", "pointerup", "mouseup", "click"].forEach((evt) => {
                clickTarget.dispatchEvent(new MouseEvent(evt, { bubbles: true, cancelable: true, composed: true }));
              });
              try { clickTarget.click?.(); } catch (_) {}
            }
            return { clicked: true, selector: found.selector, waitedMs: Date.now() - started, randomIndex: found.randomIndex, randomPoolSize: found.randomPoolSize };
          }
          await new Promise((resolve) => setTimeout(resolve, 250));
        }

        const fallbackResult = tryXFormSubmitFallback();
        if (fallbackResult.clicked) {
          return { ...fallbackResult, waitedMs: Date.now() - started };
        }

        const found = findCandidate();
        if (!found) return { clicked: false, reason: "not_found", selectors };
        if (!isEnabled(found.el)) return { clicked: false, reason: "disabled", selector: found.selector };
        if (opts.waitForUpload && !hasUploadReadySignal()) return { clicked: false, reason: "upload_not_ready", selector: found.selector };
        return { clicked: false, reason: "unknown", selector: found.selector };
      }, [candidates, (() => {
        const requestedTimeoutMs = Number(payload.timeoutMs || 0);
        const defaultTimeoutMs = msg.name === "post.submit" ? 60000 : 30000;
        const timeoutMs = Math.max(defaultTimeoutMs, requestedTimeoutMs || defaultTimeoutMs);
        return {
          waitForUpload: Boolean(payload.waitForUpload),
          timeoutMs,
          isDraft,
          platform,
          useRandomMatch,
          textContains: String(payload.textContains || "").trim().toLowerCase(),
          singleClick: Boolean(payload.singleClick),
        };
      })()], platform);
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
          const block = editable.querySelector("div.public-DraftStyleDefault-block")
            || editable.querySelector("[data-block='true'] .public-DraftStyleDefault-block")
            || editable.querySelector("[data-block='true']");
          const textSpan = block
            ? (block.querySelector("span[data-offset-key]") || block.querySelector("span[data-text='true']"))
            : (editable.querySelector("span[data-offset-key]") || editable.querySelector("span[data-text='true']"));

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
              const br = textSpan.querySelector("br[data-text='true']");
              if (br) {
                try { br.remove(); } catch (_) {}
              }
            } else if (block) {
              block.textContent = text;
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
          return String(editable.textContent || "").trim().length > 0;
        };

        const setContentEditableByPaste = (el, value) => {
          const text = String(value || "");
          el.focus();
          el.dispatchEvent(new MouseEvent("mousedown", { bubbles: true, cancelable: true, composed: true }));
          el.dispatchEvent(new MouseEvent("mouseup", { bubbles: true, cancelable: true, composed: true }));
          el.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true, composed: true }));

          let applied = false;
          try {
            const data = new DataTransfer();
            data.setData("text/plain", text);
            const pasteEvt = new ClipboardEvent("paste", { bubbles: true, cancelable: true, clipboardData: data });
            el.dispatchEvent(pasteEvt);
            applied = String(el.textContent || "").trim().length > 0;
          } catch (_) {}

          if (!applied) {
            try { document.execCommand("selectAll", false, null); } catch (_) {}
            try { applied = Boolean(document.execCommand("insertText", false, text)); } catch (_) { applied = false; }
          }

          if (!applied || String(el.textContent || "").trim().length === 0) {
            el.textContent = text;
          }

          try { el.dispatchEvent(new InputEvent("beforeinput", { bubbles: true, composed: true, data: text, inputType: "insertFromPaste" })); } catch (_) {}
          try { el.dispatchEvent(new InputEvent("input", { bubbles: true, composed: true, data: text, inputType: "insertFromPaste" })); } catch (_) {
            el.dispatchEvent(new Event("input", { bubbles: true }));
          }
          el.dispatchEvent(new Event("change", { bubbles: true }));
          el.dispatchEvent(new Event("blur", { bubbles: true }));
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
            return setContentEditableByPaste(el, value);
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

        const getEditableText = (el) => String(el?.innerText || el?.textContent || "").replace(/\u00a0/g, " ").trim();

        const clearEditable = (el) => {
          if (!el) return;
          try { el.focus(); } catch (_) {}
          try { document.execCommand("selectAll", false, null); } catch (_) {}
          try { document.execCommand("delete", false, null); } catch (_) {}
          try { el.textContent = ""; } catch (_) {}
          try { el.dispatchEvent(new Event("input", { bubbles: true })); } catch (_) {}
        };

        const typeTextIntoEditable = async (el, value) => {
          if (!el) return false;
          const text = String(value || "");
          clearEditable(el);
          for (const ch of text) {
            try { el.focus(); } catch (_) {}
            let inserted = false;
            try { inserted = Boolean(document.execCommand("insertText", false, ch)); } catch (_) { inserted = false; }
            if (!inserted) {
              try { el.textContent = `${el.textContent || ""}${ch}`; } catch (_) {}
            }
            try { el.dispatchEvent(new InputEvent("input", { bubbles: true, composed: true, data: ch, inputType: "insertText" })); } catch (_) {
              try { el.dispatchEvent(new Event("input", { bubbles: true })); } catch (_) {}
            }
            await wait(8);
          }
          try { el.dispatchEvent(new Event("change", { bubbles: true })); } catch (_) {}
          try { el.dispatchEvent(new Event("blur", { bubbles: true })); } catch (_) {}
          return getEditableText(el).length > 0;
        };

        const typeTikTokCaptionWithHashtagTabs = async (el, value) => {
          if (!el) return false;
          const text = String(value || "");
          clearEditable(el);

          const insertToken = async (token) => {
            if (!token) return;
            await typeToken(el, token);
          };

          let idx = 0;
          while (idx < text.length) {
            const ch = text[idx];
            if (ch === "#") {
              let end = idx + 1;
              while (end < text.length && /[A-Za-z0-9_\-.]/.test(text[end])) end += 1;
              const hashtag = text.slice(idx, end);
              if (hashtag.length > 1) {
                await insertToken(`${hashtag}	`);
                await wait(90);
                idx = end;
                continue;
              }
            }

            await insertToken(ch);
            idx += 1;
          }

          try { el.dispatchEvent(new Event("change", { bubbles: true })); } catch (_) {}
          try { el.dispatchEvent(new Event("blur", { bubbles: true })); } catch (_) {}
          return getEditableText(el).length > 0;
        };

        const setYouTubeRichTextboxValue = (el, value) => {
          if (!el) return false;
          const text = String(value || "");
          try { el.focus(); } catch (_) {}
          try {
            const sel = window.getSelection?.();
            const range = document.createRange();
            range.selectNodeContents(el);
            range.collapse(false);
            sel?.removeAllRanges();
            sel?.addRange(range);
          } catch (_) {}

          let inserted = false;
          try {
            document.execCommand("selectAll", false, null);
            document.execCommand("delete", false, null);
            inserted = Boolean(document.execCommand("insertText", false, text));
          } catch (_) {
            inserted = false;
          }

          if (!inserted) {
            try { el.textContent = text; } catch (_) {}
          }

          try { el.dispatchEvent(new InputEvent("beforeinput", { bubbles: true, composed: true, data: text, inputType: "insertText" })); } catch (_) {}
          try { el.dispatchEvent(new InputEvent("input", { bubbles: true, composed: true, data: text, inputType: "insertText" })); } catch (_) {
            try { el.dispatchEvent(new Event("input", { bubbles: true })); } catch (_) {}
          }
          try { el.dispatchEvent(new Event("change", { bubbles: true })); } catch (_) {}
          return true;
        };

        const setYouTubeTextboxTextContent = (el, value) => {
          if (!el) return false;
          const text = String(value || "");
          try { el.focus(); } catch (_) {}
          try { el.textContent = text; } catch (_) { return false; }
          try { el.dispatchEvent(new InputEvent("beforeinput", { bubbles: true, composed: true, data: text, inputType: "insertText" })); } catch (_) {}
          try { el.dispatchEvent(new InputEvent("input", { bubbles: true, composed: true, data: text, inputType: "insertText" })); } catch (_) {
            try { el.dispatchEvent(new Event("input", { bubbles: true })); } catch (_) {}
          }
          try { el.dispatchEvent(new Event("change", { bubbles: true })); } catch (_) {}
          return true;
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
          const typed = setValue(el, p.value ?? p.text ?? "");
          if (typed && p.submit) {
            try {
              const submitEvent = new KeyboardEvent("keydown", {
                key: "Enter",
                code: "Enter",
                keyCode: 13,
                which: 13,
                bubbles: true,
                cancelable: true,
              });
              el.dispatchEvent(submitEvent);
            } catch (_) {}
            try {
              const target = el.closest("form") || el;
              target.dispatchEvent(new KeyboardEvent("keyup", {
                key: "Enter",
                code: "Enter",
                keyCode: 13,
                which: 13,
                bubbles: true,
                cancelable: true,
              }));
            } catch (_) {}
          }
          return { typed, selector: p.selector || "", submitted: Boolean(p.submit) };
        }

        const fields = p.fields || {};
        const focusYouTubeTextbox = (textbox) => {
          if (!textbox) return null;
          const outer = textbox.closest("#outer, #child-input, #container-content") || textbox.parentElement;
          if (outer) {
            try { outer.scrollIntoView({ block: "center", inline: "center" }); } catch (_) {}
          }
          // Avoid synthetic mouse clicks here: YouTube may call
          // requestStorageAccessFor() in click handlers which requires a trusted
          // user gesture and logs console errors when triggered programmatically.
          try { textbox.focus(); } catch (_) {}
          return textbox;
        };

        const findFacebookReelDescriptionField = () => {
          const selectors = [
            "div[contenteditable='true'][role='textbox'][data-lexical-editor='true'][aria-placeholder*='Describe your reel' i]",
            "div[contenteditable='true'][data-lexical-editor='true'][aria-placeholder*='Describe your reel' i]",
            "div[contenteditable='true'][role='textbox'][aria-placeholder*='Describe your reel' i]",
            "div[contenteditable='true'][data-lexical-editor='true']",
          ];
          for (const sel of selectors) {
            const el = document.querySelector(sel);
            if (!el) continue;
            try { el.scrollIntoView({ block: "center", inline: "center" }); } catch (_) {}
            try { el.dispatchEvent(new MouseEvent("mousedown", { bubbles: true, cancelable: true, composed: true })); } catch (_) {}
            try { el.dispatchEvent(new MouseEvent("mouseup", { bubbles: true, cancelable: true, composed: true })); } catch (_) {}
            try { el.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true, composed: true })); } catch (_) {}
            try { el.focus(); } catch (_) {}
            return el;
          }
          return null;
        };

        const findXComposerField = () => {
          const selectors = [
            "div[data-testid='tweetTextarea_0'][contenteditable='true']",
            "div[data-testid^='tweetTextarea'][contenteditable='true']",
            "div[role='textbox'][contenteditable='true'][aria-label*='post text' i]",
          ];

          const isVisible = (el) => Boolean(el && (el.offsetWidth || el.offsetHeight || el.getClientRects().length));

          for (const sel of selectors) {
            const editable = Array.from(document.querySelectorAll(sel)).find((node) => isVisible(node))
              || document.querySelector(sel);
            if (!editable) continue;
            try { editable.scrollIntoView({ block: "center", inline: "center" }); } catch (_) {}
            try { editable.dispatchEvent(new MouseEvent("mousedown", { bubbles: true, cancelable: true, composed: true })); } catch (_) {}
            try { editable.dispatchEvent(new MouseEvent("mouseup", { bubbles: true, cancelable: true, composed: true })); } catch (_) {}
            try { editable.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true, composed: true })); } catch (_) {}
            try { editable.focus(); } catch (_) {}
            return editable;
          }

          return null;
        };

        const findYouTubeContainerField = (key) => {
          const titleCandidates = [
            "#textbox[contenteditable='true'][aria-label*='add a title' i]",
            "#textbox[contenteditable='true'][aria-required='true'][aria-label*='title' i]",
            "#title-textarea #textbox[contenteditable='true']",
          ];
          const descriptionCandidates = [
            "#textbox[contenteditable='true'][aria-label*='tell viewers about your video' i]",
            "#description #textbox[contenteditable='true']",
            "#textbox[contenteditable='true'][aria-label*='description' i]",
          ];

          if (key === "title") {
            for (const sel of titleCandidates) {
              const found = document.querySelector(sel);
              if (found) return focusYouTubeTextbox(found);
            }
          }

          if (key === "description") {
            for (const sel of descriptionCandidates) {
              const found = document.querySelector(sel);
              if (found) return focusYouTubeTextbox(found);
            }
          }

          const containers = Array.from(document.querySelectorAll("ytcp-form-input-container#container, ytcp-form-input-container"));
          for (const container of containers) {
            const root = container.closest("ytcp-form-input-container") || container;
            const label = String(root.querySelector("#label-text")?.textContent || "").toLowerCase();
            if (key === "title" && !label.includes("title")) continue;
            if (key === "description" && !label.includes("description")) continue;
            const textbox = root.querySelector("#textbox[contenteditable='true'], textarea#textbox");
            if (textbox) return focusYouTubeTextbox(textbox);
          }
          return null;
        };

        const selectors = {
          title: [
            "#title-textarea #textbox[contenteditable='true']",
            "ytcp-form-input-container #outer #textbox[contenteditable='true']",
            "div#textbox[contenteditable='true'][aria-label*='title' i]",
            "textarea#title-textarea",
            "#textbox",
            "input[name='title']",
            "textarea[name='title']"
          ],
          description: [
            "div[contenteditable='true'][role='textbox'][data-lexical-editor='true'][aria-placeholder*='Describe your reel' i]",
            "div[contenteditable='true'][data-lexical-editor='true'][aria-placeholder*='Describe your reel' i]",
            "div[contenteditable='true'][role='textbox'][aria-placeholder*='Describe your reel' i]",
            "#description #textbox[contenteditable='true']",
            "div#textbox[contenteditable='true'][aria-label*='tell viewers' i]",
            "div#textbox[contenteditable='true'][aria-label*='description' i]",
            "[contenteditable='true'][aria-placeholder*='describe your reel' i]",
            ".DraftEditor-editorContainer [contenteditable='true'][role='combobox']",
            "div.public-DraftEditor-content[contenteditable='true'][role='combobox']",
            "div[contenteditable='true'][role='combobox']",
            "div[contenteditable='true'][role='textbox']",
            "div[role='textbox'][contenteditable='true']",
            "div[data-testid='tweetTextarea_0'][contenteditable='true']",
            "textarea[aria-label*='Write a caption' i]",
            "div[contenteditable='true']",
            "textarea#description-textarea",
            "textarea[name='description']"
          ]
        };
        const fieldAlias = {
          caption: "description",
          desc: "description",
          text: "description",
          body: "description",
        };

        const out = {};
        for (const [rawKey, value] of Object.entries(fields)) {
          const key = String(rawKey || "").toLowerCase();
          const canonicalKey = fieldAlias[key] || key;
          const list = selectors[canonicalKey] || selectors[key] || [`[name='${key}']`, `textarea[name='${key}']`, `input[name='${key}']`];
          let el = null;
          if (currentPlatform === "youtube" && (canonicalKey === "title" || canonicalKey === "description")) {
            el = findYouTubeContainerField(canonicalKey);
          }
          if (!el) {
            el = list.map((sel) => document.querySelector(sel)).find(Boolean);
          }
          if (!el) {
            out[rawKey] = false;
            continue;
          }

          const text = String(value || "");
          if (currentPlatform === "tiktok" && canonicalKey === "description") {
            const tiktokSelectors = [
              ".DraftEditor-editorContainer [contenteditable='true'][role='combobox']",
              "div.public-DraftEditor-content[contenteditable='true'][role='combobox']",
              "div[contenteditable='true'][role='combobox']",
              "div[contenteditable='true']",
            ];
            const tiktokEl = tiktokSelectors.map((sel) => document.querySelector(sel)).find(Boolean) || el;
            const hasHashtags = /(^|\s)#[A-Za-z0-9_\-.]+/.test(text);
            if (hasHashtags) {
              out[rawKey] = await typeTikTokCaptionWithHashtagTabs(tiktokEl, text);
            } else {
              const pasteOk = setValue(tiktokEl, text);
              const matches = getEditableText(tiktokEl).includes(text.trim());
              out[rawKey] = matches ? pasteOk : await typeTextIntoEditable(tiktokEl, text);
            }
          } else if (currentPlatform === "facebook" && canonicalKey === "description") {
            const facebookEl = findFacebookReelDescriptionField() || el;
            out[rawKey] = setValue(facebookEl, text);
          } else if (currentPlatform === "x" && canonicalKey === "description") {
            const xEl = findXComposerField() || el;
            const normalizeForCompare = (value) => String(value || "")
              .replace(/\u200B/g, "")
              .replace(/\s+/g, " ")
              .trim();
            const expected = normalizeForCompare(text);
            const draftTextMatches = () => {
              const current = normalizeForCompare(getEditableText(xEl));
              if (!current || !expected) return false;
              return current === expected || current.includes(expected) || expected.includes(current);
            };

            const filled = draftTextMatches()
              || setValue(xEl, text)
              || await typeTextIntoEditable(xEl, text);
            out[rawKey] = Boolean(filled) && draftTextMatches();
          } else if (currentPlatform === "instagram" && canonicalKey === "description") {
            out[rawKey] = setValue(el, text);
          } else if (currentPlatform === "youtube" && (canonicalKey === "title" || canonicalKey === "description")) {
            const normalizeForCompare = (value) => String(value || "")
              .replace(/\u200B/g, "")
              .replace(/\s+/g, " ")
              .trim();
            const expected = normalizeForCompare(text);
            const youtubeTextMatches = () => {
              const current = normalizeForCompare(getEditableText(el));
              if (!expected) return current.length === 0;
              return current === expected || current.includes(expected) || expected.includes(current);
            };

            const filled = youtubeTextMatches()
              || setYouTubeRichTextboxValue(el, text)
              || setYouTubeTextboxTextContent(el, text);
            out[rawKey] = Boolean(filled) && youtubeTextMatches();
          } else {
            out[rawKey] = setValue(el, text);
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
      const platform = String(payload.platform || "").toLowerCase();
      const filePath = String(payload.filePath || "");
      if (!filePath) {
        ackCmd(msg, false, {}, "filePath is required");
        return;
      }
      const tab = await findTargetTab(platform);
      const debuggee = { tabId: tab.id };
      const selectorExpr = `(() => { const nodes = Array.from(document.querySelectorAll("input[type=\'file\']")); return nodes.find((n) => (n.offsetWidth || n.offsetHeight || n.getClientRects().length)) || nodes[0] || null; })()`;
      let attached = false;
      try {
        await chrome.debugger.attach(debuggee, "1.3");
        attached = true;
        await chrome.debugger.sendCommand(debuggee, "DOM.enable", {});
        await chrome.debugger.sendCommand(debuggee, "Runtime.enable", {});
        const evalRes = await chrome.debugger.sendCommand(debuggee, "Runtime.evaluate", {
          expression: selectorExpr,
          returnByValue: false,
        });
        const objectId = evalRes?.result?.objectId;
        if (!objectId) throw new Error("file input element not found in target tab");
        const nodeInfo = await chrome.debugger.sendCommand(debuggee, "DOM.requestNode", { objectId });
        const nodeId = nodeInfo?.nodeId;
        if (!nodeId) throw new Error("failed to resolve file input nodeId");
        await chrome.debugger.sendCommand(debuggee, "DOM.setFileInputFiles", { nodeId, files: [filePath] });
        sendEvent("state", { state: "upload_selected", platform, filePath, mode: "extension_debugger_set_file_input_files" });
        ackCmd(msg, true, { mode: "extension_debugger_set_file_input_files" });
      } catch (err) {
        ackCmd(msg, false, {}, err?.message || String(err));
      } finally {
        if (attached) {
          try { await chrome.debugger.detach(debuggee); } catch (_) {}
        }
      }
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
