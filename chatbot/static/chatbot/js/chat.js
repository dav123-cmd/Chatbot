(() => {
  const thread = document.getElementById("thread");
  const form = document.getElementById("composer");
  const input = document.getElementById("user-input");
  const sendBtn = document.getElementById("send-btn");
  const statusEl = document.getElementById("status");
  const statusLabel = document.getElementById("status-label");
  const metaText = document.getElementById("meta-text");

  const CHAT_ENDPOINT = "/api/chat/";

  function setStatus(state, label) {
    statusEl.classList.remove("online", "error");
    if (state) statusEl.classList.add(state);
    statusLabel.textContent = label;
  }

  function scrollToBottom() {
    thread.scrollTop = thread.scrollHeight;
  }

  function appendMessage(text, sender) {
    const row = document.createElement("div");
    row.className = `msg msg--${sender}`;
    const bubble = document.createElement("div");
    bubble.className = "msg__bubble";
    bubble.textContent = text;
    row.appendChild(bubble);
    thread.appendChild(row);
    scrollToBottom();
    return row;
  }

  function appendThinking() {
    const row = document.createElement("div");
    row.className = "msg msg--bot msg--thinking";
    row.innerHTML = `
      <div class="msg__bubble">
        <span class="dot"></span><span class="dot"></span><span class="dot"></span>
      </div>`;
    thread.appendChild(row);
    scrollToBottom();
    return row;
  }

  function getCookie(name) {
    const match = document.cookie.match(new RegExp(`(^| )${name}=([^;]+)`));
    return match ? decodeURIComponent(match[2]) : null;
  }

  async function sendMessage(message) {
    appendMessage(message, "user");
    input.value = "";
    sendBtn.disabled = true;

    const thinkingRow = appendThinking();

    try {
      const response = await fetch(CHAT_ENDPOINT, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-CSRFToken": getCookie("csrftoken") || "",
        },
        body: JSON.stringify({ message }),
      });

      const data = await response.json();
      thinkingRow.remove();

      if (!response.ok) {
        setStatus("error", "model unavailable");
        appendMessage(
          data.error || "Something went wrong talking to the model.",
          "bot"
        );
        return;
      }

      setStatus("online", "online");
      appendMessage(data.response, "bot");
      metaText.textContent = `last intent: ${data.intent} · confidence: ${(data.confidence * 100).toFixed(1)}%`;
    } catch (err) {
      thinkingRow.remove();
      setStatus("error", "connection error");
      appendMessage("I couldn't reach the server. Please try again.", "bot");
    } finally {
      sendBtn.disabled = false;
      input.focus();
    }
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const message = input.value.trim();
    if (!message) return;
    sendMessage(message);
  });

  setStatus(null, "ready");
  input.focus();
})();
