"""
Giveaway Review Console — PySide6 (Qt) Full App
----------------------------------------------
Desktop app to search recent X posts and (on your explicit click) Like / Retweet /
Follow / Reply. Uses Tweepy v2 with optional app-context (Bearer) for search.

Key features
- Search with optional "New only" mode using since_id (persists to .giveaway_state.json)
- Exponential backoff on 429 with jitter
- STOP button to cancel long scans (paginates and checks stop flag between calls)
- Manual actions: Like, Retweet, Follow, Reply (with adjustable random delay)
- Heuristic: detects posts that ask the user to follow (safe regex + self-tests)

Setup
  pip install PySide6 tweepy python-dotenv
  Create .env alongside this file with:
    X_CONSUMER_KEY=...
    X_CONSUMER_SECRET=...
    X_ACCESS_TOKEN=...
    X_ACCESS_TOKEN_SECRET=...
    # Optional (recommended for app-context search):
    X_BEARER_TOKEN=...
  Run: python giveaway_qt.py
"""
from __future__ import annotations
import os
import re
import sys
import time
import json
import random
import webbrowser
from dataclasses import dataclass
from typing import List, Optional, Tuple

from dotenv import load_dotenv
import tweepy

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QTextEdit, QListWidget, QListWidgetItem,
    QPushButton, QSpinBox, QDoubleSpinBox, QCheckBox, QSplitter,
    QMessageBox, QStatusBar
)

# ----------------------- Auth & Client -----------------------
load_dotenv()
BEARER = os.getenv("X_BEARER_TOKEN")
CK = os.getenv("X_CONSUMER_KEY")
CS = os.getenv("X_CONSUMER_SECRET")
AT = os.getenv("X_ACCESS_TOKEN")
AS = os.getenv("X_ACCESS_TOKEN_SECRET")

client = tweepy.Client(
    bearer_token=BEARER,  # optional but recommended for app-context search
    consumer_key=CK,
    consumer_secret=CS,
    access_token=AT,
    access_token_secret=AS,
    wait_on_rate_limit=True,
)

# ----------------------- Persistence -----------------------
STATE_PATH = os.path.join(os.path.dirname(__file__), ".giveaway_state.json")

def load_state() -> dict:
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(d: dict) -> None:
    try:
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(d, f)
    except Exception:
        pass

STATE = load_state()  # e.g., {"since_id": "1870..."}

# ----------------------- Regex & Tests -----------------------
# Avoid stray non-printable backspace (\x08). Use proper word boundaries.
FOLLOW_HINTS = re.compile(r"\bfollow(?!-)\b( me| us| both| all)?\b", re.I)

def asks_to_follow(text: Optional[str]) -> bool:
    if not text:
        return False
    return bool(FOLLOW_HINTS.search(text))

SELF_TESTS = [
    ("Please follow & RT!", True),
    ("FOLLOW both accounts and RT", True),
    ("like + RT (no follow mentioned)", False),
    ("This is a follow-up announcement", False),
    ("Unfollow scams, stay safe", False),
    ("follow us for a chance to win", True),
    ("rt + like, then drop your $SOL address", False),
    (None, False),
]

# ----------------------- Data Models -----------------------
@dataclass
class Author:
    id: str
    username: str
    name: Optional[str]
    avatar_url: Optional[str]

@dataclass
class TweetRow:
    id: str
    text: str
    created_at: Optional[str]
    author: Author

    @property
    def url(self) -> str:
        return f"https://x.com/{self.author.username}/status/{self.id}"

    @property
    def label(self) -> str:
        t = (self.text or "").replace("\n", " ")
        if len(t) > 90:
            t = t[:87] + "..."
        flag = " [asks follow]" if asks_to_follow(self.text) else ""
        return f"@{self.author.username} — {t}{flag}"

# ----------------------- API Helpers -----------------------

def map_users(res) -> dict:
    return {u.id: u for u in (res.includes.get("users") or [])}


def page_search_recent(query: str, page_size: int, since_id: Optional[str], next_token: Optional[str]) -> Tuple[List[TweetRow], Optional[str], Optional[str]]:
    """Fetch one page of recent search. Returns (rows, top_id, next_token)."""
    kwargs = dict(
        query=query,
        max_results=max(10, min(int(page_size or 20), 100)),
        tweet_fields=["created_at"],
        expansions=["author_id"],
        user_fields=["username", "name", "profile_image_url"],
        sort_order="recency",
    )
    if since_id:
        kwargs["since_id"] = since_id
    if next_token:
        kwargs["next_token"] = next_token
    if not BEARER:
        kwargs["user_auth"] = True  # force user context if no bearer

    res = client.search_recent_tweets(**kwargs)
    data = res.data or []
    users = map_users(res)
    rows: List[TweetRow] = []
    top_id = since_id

    for t in data:
        u = users.get(t.author_id)
        author = Author(
            id=str(t.author_id if not u else u.id),
            username="unknown" if not u else u.username,
            name=None if not u else u.name,
            avatar_url=None if not u else getattr(u, "profile_image_url", None),
        )
        rows.append(
            TweetRow(
                id=str(t.id),
                text=t.text,
                created_at=(str(getattr(t, "created_at", "")) or None),
                author=author,
            )
        )
        if top_id is None or int(t.id) > int(top_id):
            top_id = str(t.id)

    next_tok = None
    meta = getattr(res, "meta", None)
    if meta and isinstance(meta, dict):
        next_tok = meta.get("next_token")
    return rows, top_id, next_tok


def with_delay(min_s: float, max_s: float):
    if min_s < 0: min_s = 0
    if max_s < min_s: max_s = min_s
    time.sleep(random.uniform(min_s, max_s))


def do_like(tweet_id: str, min_s: float, max_s: float):
    with_delay(min_s, max_s)
    client.like(tweet_id)


def do_retweet(tweet_id: str, min_s: float, max_s: float):
    with_delay(min_s, max_s)
    client.retweet(tweet_id)


def do_follow(user_id: str, min_s: float, max_s: float):
    with_delay(min_s, max_s)
    client.follow_user(target_user_id=user_id)


def do_reply(tweet_id: str, text: str, min_s: float, max_s: float):
    if not (text and text.strip()):
        raise ValueError("Reply text cannot be empty")
    with_delay(min_s, max_s)
    client.create_tweet(text=text.strip(), in_reply_to_tweet_id=tweet_id)

# ----------------------- Worker Threads -----------------------
class SearchWorker(QThread):
    progressed = Signal(int)  # number of rows fetched so far
    finished = Signal(list, str, object)  # rows, error, new_since_id

    def __init__(self, query: str, total_limit: int, new_only: bool, page_size: int, since_id: Optional[str]):
        super().__init__()
        self.query = query
        self.total_limit = max(1, min(int(total_limit or 20), 500))
        self.new_only = new_only
        self.page_size = max(10, min(int(page_size or 50), 100))
        self.since_id = since_id if new_only else None
        self._stop = False

    def request_stop(self):
        self._stop = True

    def backoff(self, attempt: int):
        delay = min(15 * (2 ** attempt), 300) + random.uniform(0, 3)
        for _ in range(int(delay // 0.1)):
            if self._stop:
                return
            time.sleep(0.1)

    def run(self):
        rows: List[TweetRow] = []
        next_token = None
        top_id = self.since_id
        attempt = 0
        try:
            while not self._stop and len(rows) < self.total_limit:
                try:
                    page_rows, top_id, next_token = page_search_recent(
                        self.query,
                        page_size=self.page_size,
                        since_id=top_id if self.new_only else None,
                        next_token=next_token,
                    )
                    attempt = 0  # reset on success
                except tweepy.TooManyRequests:
                    if self._stop:
                        break
                    self.backoff(attempt)
                    attempt += 1
                    if attempt > 6:
                        self.finished.emit([], "429 Rate limit reached. Try later.", None)
                        return
                    else:
                        continue
                except tweepy.Unauthorized:
                    self.finished.emit([], "401 Unauthorized. Add X_BEARER_TOKEN or ensure tweet.read scope.", None)
                    return
                except tweepy.Forbidden as e:
                    self.finished.emit([], f"403 Forbidden: {e}", None)
                    return
                except Exception as e:
                    self.finished.emit([], f"Search failed: {e}", None)
                    return

                if not page_rows:
                    break

                for r in page_rows:
                    rows.append(r)
                    if len(rows) >= self.total_limit:
                        break
                self.progressed.emit(len(rows))

                if not next_token:
                    break
            # done
            self.finished.emit(rows, "", top_id if self.new_only else (top_id or self.since_id))
        except Exception as e:
            self.finished.emit([], f"Unexpected error: {e}", None)


class ActionWorker(QThread):
    finished = Signal(str)  # message (empty if ok)
    def __init__(self, kind: str, **kwargs):
        super().__init__()
        self.kind = kind
        self.kwargs = kwargs
    def run(self):
        try:
            if self.kind == "like":
                do_like(self.kwargs["tweet_id"], self.kwargs["min_s"], self.kwargs["max_s"])
            elif self.kind == "retweet":
                do_retweet(self.kwargs["tweet_id"], self.kwargs["min_s"], self.kwargs["max_s"])
            elif self.kind == "follow":
                do_follow(self.kwargs["user_id"], self.kwargs["min_s"], self.kwargs["max_s"])
            elif self.kind == "reply":
                do_reply(self.kwargs["tweet_id"], self.kwargs["text"], self.kwargs["min_s"], self.kwargs["max_s"])
            else:
                raise ValueError("Unknown action")
            self.finished.emit("")
        except tweepy.TooManyRequests:
            self.finished.emit("429 Rate limit reached. Please slow down.")
        except tweepy.Unauthorized as e:
            self.finished.emit("401 Unauthorized — check tokens/scopes or add X_BEARER_TOKEN.")
        except tweepy.Forbidden as e:
            self.finished.emit("403 Forbidden — missing write permission on your app?")
        except Exception as e:
            self.finished.emit(f"Unexpected error: {e}")

# ----------------------- Main Window -----------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Giveaway Review Console (PySide6)")

        self.results: List[TweetRow] = []
        self.selected_index: Optional[int] = None
        self.search_worker: Optional[SearchWorker] = None

        # Left controls
        self.q_edit = QTextEdit();
        self.q_edit.setPlainText('("follow" OR "rt" OR "retweet") ("drop your $SOL address" OR "$SOL address" OR "sol address") lang:en -is:retweet')
        self.limit_spin = QSpinBox(); self.limit_spin.setRange(10, 500); self.limit_spin.setValue(60)
        self.page_spin = QSpinBox(); self.page_spin.setRange(10, 100); self.page_spin.setValue(50)
        self.new_only_chk = QCheckBox("New only (since last scan)"); self.new_only_chk.setChecked(True)
        self.scan_btn = QPushButton("Scan")
        self.stop_btn = QPushButton("Stop"); self.stop_btn.setEnabled(False)
        self.test_btn = QPushButton("Run Tests")

        self.sol_edit = QLineEdit(); self.sol_edit.setPlaceholderText("Your SOL address")
        self.auto_chk = QCheckBox("Auto-build reply from SOL address"); self.auto_chk.setChecked(True)
        self.reply_edit = QTextEdit()
        self.min_spin = QDoubleSpinBox(); self.min_spin.setRange(0, 999); self.min_spin.setDecimals(1); self.min_spin.setValue(3)
        self.max_spin = QDoubleSpinBox(); self.max_spin.setRange(0, 999); self.max_spin.setDecimals(1); self.max_spin.setValue(9)

        left = QWidget(); lv = QVBoxLayout(left)
        row1 = QHBoxLayout(); row1.addWidget(QLabel("Search")); row1.addStretch(); row1.addWidget(self.test_btn)
        lv.addLayout(row1)
        lv.addWidget(QLabel("Query")); lv.addWidget(self.q_edit)
        row2 = QHBoxLayout();
        row2.addWidget(QLabel("Total limit")); row2.addWidget(self.limit_spin)
        row2.addWidget(QLabel("Page size")); row2.addWidget(self.page_spin)
        row2.addStretch(); row2.addWidget(self.new_only_chk)
        lv.addLayout(row2)
        rowScan = QHBoxLayout(); rowScan.addWidget(self.scan_btn); rowScan.addWidget(self.stop_btn); lv.addLayout(rowScan)

        lv.addWidget(QLabel("Engagement Settings"))
        lv.addWidget(QLabel("Your SOL address")); lv.addWidget(self.sol_edit)
        lv.addWidget(self.auto_chk)
        lv.addWidget(QLabel("Reply text")); lv.addWidget(self.reply_edit)
        row3 = QHBoxLayout(); row3.addWidget(QLabel("Delay (s) min")); row3.addWidget(self.min_spin); row3.addWidget(QLabel("max")); row3.addWidget(self.max_spin); row3.addStretch()
        lv.addLayout(row3)

        # Right results
        self.listbox = QListWidget()
        self.detail = QTextEdit(); self.detail.setReadOnly(True)
        self.like_btn = QPushButton("Like")
        self.rt_btn = QPushButton("Retweet")
        self.follow_btn = QPushButton("Follow")
        self.reply_btn = QPushButton("Reply")
        self.open_btn = QPushButton("Open in browser")

        right = QWidget(); rv = QVBoxLayout(right)
        rv.addWidget(QLabel("Results"))
        rv.addWidget(self.listbox)
        rv.addWidget(self.detail)
        row4 = QHBoxLayout(); [row4.addWidget(b) for b in (self.like_btn, self.rt_btn, self.follow_btn, self.reply_btn, self.open_btn)]
        rv.addLayout(row4)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        container = QWidget(); layout = QVBoxLayout(container); layout.addWidget(splitter)
        self.setCentralWidget(container)
        self.setStatusBar(QStatusBar())

        # Signals
        self.scan_btn.clicked.connect(self.on_scan)
        self.stop_btn.clicked.connect(self.on_stop)
        self.test_btn.clicked.connect(self.on_tests)
        self.listbox.currentRowChanged.connect(self.on_select)
        self.open_btn.clicked.connect(self.on_open)
        self.like_btn.clicked.connect(lambda: self.run_action("like"))
        self.rt_btn.clicked.connect(lambda: self.run_action("retweet"))
        self.follow_btn.clicked.connect(lambda: self.run_action("follow"))
        self.reply_btn.clicked.connect(lambda: self.run_action("reply"))
        self.sol_edit.textChanged.connect(self.sync_reply)
        self.auto_chk.toggled.connect(self.sync_reply)
        self.sync_reply()

        # Show creds status
        self.statusBar().showMessage(self.creds_summary())

    # ---------- Helpers ----------
    def creds_summary(self) -> str:
        return f"bearer={'Y' if BEARER else 'N'} userKeys={'Y' if (CK and CS) else 'N'} userTokens={'Y' if (AT and AS) else 'N'}"

    def info(self, msg: str):
        self.detail.setPlainText(msg)
        self.statusBar().showMessage(msg[:120])

    def warn(self, title: str, msg: str):
        QMessageBox.warning(self, title, msg)

    def error(self, title: str, msg: str):
        QMessageBox.critical(self, title, msg)

    def sync_reply(self):
        if self.auto_chk.isChecked():
            sol = self.sol_edit.text().strip()
            self.reply_edit.setPlainText(f"Here’s my SOL address: {sol}" if sol else "")

    # ---------- Events ----------
    def on_tests(self):
        passed = 0
        lines = []
        for text, expected in SELF_TESTS:
            got = asks_to_follow(text)
            ok = got == expected
            passed += int(ok)
            lines.append(f"{'✓' if ok else '✗'} expected={expected} got={got} | {text}")
        creds = self.creds_summary()
        QMessageBox.information(self, "Self‑test", f"Self‑test {passed}/{len(SELF_TESTS)} passed\n\n" + "\n".join(lines) + f"\n\nCreds: {creds}")

    def on_scan(self):
        if self.search_worker and self.search_worker.isRunning():
            self.warn("Busy", "A scan is already running")
            return
        q = self.q_edit.toPlainText().strip()
        total_limit = int(self.limit_spin.value())
        page_size = int(self.page_spin.value())
        new_only = self.new_only_chk.isChecked()
        since_id = STATE.get("since_id") if new_only else None
        self.info("Scanning… (Stop available)")
        self.scan_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.search_worker = SearchWorker(q, total_limit, new_only, page_size, since_id)
        self.search_worker.progressed.connect(self.on_progress)
        self.search_worker.finished.connect(self.on_scan_done)
        self.search_worker.start()

    def on_progress(self, n: int):
        self.statusBar().showMessage(f"Fetched {n} posts… (Stop available)")

    def on_stop(self):
        if self.search_worker and self.search_worker.isRunning():
            self.search_worker.request_stop()
            self.statusBar().showMessage("Stopping scan…")

    def on_scan_done(self, rows: list, err: str, new_since_id: Optional[str]):
        self.scan_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if err:
            self.info(err)
            return
        # Prepend new rows in New-only mode, otherwise replace
        if self.new_only_chk.isChecked():
            self.results = rows + self.results
        else:
            self.results = rows
        self.listbox.clear()
        for r in self.results:
            self.listbox.addItem(QListWidgetItem(r.label))
        self.info(f"Found {len(rows)} posts. Total listed: {len(self.results)}")
        if self.new_only_chk.isChecked() and new_since_id:
            STATE["since_id"] = new_since_id
            save_state(STATE)

    def on_select(self, idx: int):
        self.selected_index = idx if idx >= 0 else None
        if self.selected_index is None:
            return
        t = self.results[self.selected_index]
        flags = "\n[asks to follow]" if asks_to_follow(t.text) else ""
        self.detail.setPlainText(f"@{t.author.username} — {t.created_at}\n{t.url}{flags}\n\n{t.text}")

    def on_open(self):
        if self.selected_index is None:
            self.warn("No selection", "Select a post first")
            return
        webbrowser.open(self.results[self.selected_index].url)

    def run_action(self, kind: str):
        if self.selected_index is None:
            self.warn("No selection", "Select a post first")
            return
        t = self.results[self.selected_index]
        min_s = float(self.min_spin.value()); max_s = float(self.max_spin.value())
        kwargs = {"min_s": min_s, "max_s": max_s}
        if kind in ("like", "retweet"):
            kwargs["tweet_id"] = t.id
        elif kind == "follow":
            kwargs["user_id"] = t.author.id
        elif kind == "reply":
            text = self.reply_edit.toPlainText().strip()
            if not text:
                self.warn("Empty reply", "Please enter reply text")
                return
            kwargs["tweet_id"] = t.id
            kwargs["text"] = text
        else:
            self.error("Unknown action", kind)
            return
        self.disable_actions(True)
        self.action_worker = ActionWorker(kind, **kwargs)
        self.action_worker.finished.connect(self.on_action_done)
        self.action_worker.start()

    def on_action_done(self, msg: str):
        self.disable_actions(False)
        if msg:
            self.error("Action failed", msg)
        else:
            QMessageBox.information(self, "Done", "Action completed ✓")

    def disable_actions(self, disabled: bool):
        for b in (self.like_btn, self.rt_btn, self.follow_btn, self.reply_btn):
            b.setDisabled(disabled)

# ----------------------- Entry -----------------------
def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1200, 750)
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
