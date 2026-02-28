const state = {
  backendUrl: "http://127.0.0.1:8001",
  socket: null,
  mediaStream: null,
  sendTimer: null,
  processingIdleTimer: null,
  reconnectTimer: null,
  reconnectAttempts: 0,
  cameraRunning: false,
  frameInFlight: false,
  frameInFlightStartedAt: 0,
  processingIdleDelayMs: 350,
  processingMinBusyMs: 320,
  ttsEnabled: false,
  debugVisible: false,
  sendFps: 15,
  frameSize: 416,
  minConfidence: 0.45,
  unknownLabel: "unknown",
  modelClasses: [],
  signSeqTokens: [],
  signSeqIndex: -1,
  signSeqPlaying: false,
  signSeqTimer: null,
  signSeqFrameMs: 700,
}

const ui = {
  backendUrl: document.getElementById("backend-url"),
  saveBackendBtn: document.getElementById("save-backend-btn"),
  cameraStatus: document.getElementById("camera-status"),
  socketStatus: document.getElementById("socket-status"),
  modelStatus: document.getElementById("model-status"),
  processingStatus: document.getElementById("processing-status"),
  webcam: document.getElementById("webcam"),
  startCameraBtn: document.getElementById("start-camera-btn"),
  stopCameraBtn: document.getElementById("stop-camera-btn"),
  rawSign: document.getElementById("raw-sign"),
  stableSign: document.getElementById("stable-sign"),
  stableSignPill: document.getElementById("stable-sign-pill"),
  confidence: document.getElementById("confidence"),
  modelFps: document.getElementById("model-fps"),
  latency: document.getElementById("latency"),
  networkLatency: document.getElementById("network-latency"),
  committedText: document.getElementById("committed-text"),
  backspaceBtn: document.getElementById("backspace-btn"),
  clearTextBtn: document.getElementById("clear-text-btn"),
  ttsToggleBtn: document.getElementById("tts-toggle-btn"),
  debugToggleBtn: document.getElementById("debug-toggle-btn"),
  debugPanel: document.getElementById("debug-panel"),
  topkList: document.getElementById("topk-list"),
  bufferList: document.getElementById("buffer-list"),
  errorMessage: document.getElementById("error-message"),
  commitFlashTarget: document.getElementById("commit-flash-target"),
  textToSignInput: document.getElementById("text-to-sign-input"),
  useAssembledBtn: document.getElementById("use-assembled-btn"),
  generateSignSeqBtn: document.getElementById("generate-sign-seq-btn"),
  playSignSeqBtn: document.getElementById("play-sign-seq-btn"),
  stopSignSeqBtn: document.getElementById("stop-sign-seq-btn"),
  signPreviewImage: document.getElementById("sign-preview-image"),
  signPreviewFallback: document.getElementById("sign-preview-fallback"),
  signPreviewLabel: document.getElementById("sign-preview-label"),
  signPreviewStatus: document.getElementById("sign-preview-status"),
  signTokenTimeline: document.getElementById("sign-token-timeline"),
}

const sendCanvas = document.createElement("canvas")
const sendCtx = sendCanvas.getContext("2d")

function loadBackendUrl() {
  const saved = localStorage.getItem("isl_backend_url")
  if (saved) {
    state.backendUrl = saved
  }
  ui.backendUrl.value = state.backendUrl
}

function setStatusChip(element, text, kind) {
  element.textContent = text
  element.classList.remove("status-good", "status-warn", "status-bad")
  if (kind) {
    element.classList.add(kind)
  }
}

function setError(message) {
  ui.errorMessage.textContent = message || "None"
}

function setProcessingStatus(isProcessing) {
  if (isProcessing) {
    setStatusChip(ui.processingStatus, "Processing: in-flight", "status-good")
    return
  }
  setStatusChip(ui.processingStatus, "Processing: idle", "status-warn")
}

function clearProcessingIdleTimer() {
  if (!state.processingIdleTimer) {
    return
  }
  clearTimeout(state.processingIdleTimer)
  state.processingIdleTimer = null
}

function markFrameInFlight() {
  clearProcessingIdleTimer()
  state.frameInFlightStartedAt = Date.now()
  state.frameInFlight = true
  setProcessingStatus(true)
}

function resetFrameInFlight(immediate = false) {
  state.frameInFlight = false
  clearProcessingIdleTimer()

  if (immediate) {
    state.frameInFlightStartedAt = 0
    setProcessingStatus(false)
    return
  }

  const elapsed = state.frameInFlightStartedAt ? Date.now() - state.frameInFlightStartedAt : 0
  const holdMs = Math.max(state.processingIdleDelayMs, state.processingMinBusyMs - elapsed, 0)
  state.processingIdleTimer = setTimeout(() => {
    if (!state.frameInFlight) {
      setProcessingStatus(false)
    }
    state.frameInFlightStartedAt = 0
  }, holdMs)
}

function asWsUrl(httpUrl) {
  const url = new URL(httpUrl)
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:"
  url.pathname = "/ws"
  url.search = ""
  url.hash = ""
  return url.toString()
}

async function refreshHealth() {
  try {
    const res = await fetch(`${state.backendUrl}/health`, { cache: "no-store" })
    if (!res.ok) {
      throw new Error(`Health check failed (${res.status})`)
    }

    const data = await res.json()
    if (typeof data.send_fps === "number" && data.send_fps > 0) {
      state.sendFps = data.send_fps
      if (state.cameraRunning) {
        startSendLoop()
      }
    }
    if (typeof data.frame_size === "number" && data.frame_size > 0) {
      state.frameSize = data.frame_size
    }
    if (typeof data.min_confidence === "number" && data.min_confidence > 0) {
      state.minConfidence = data.min_confidence
    }
    if (typeof data.unknown_label === "string" && data.unknown_label.length > 0) {
      state.unknownLabel = data.unknown_label
    }
    if (Array.isArray(data.classes)) {
      state.modelClasses = data.classes.map((value) => String(value))
    }

    if (data.model_loaded) {
      setStatusChip(ui.modelStatus, `Model: loaded (${data.mode})`, "status-good")
      if (data.hand_roi_enabled && data.hand_roi_available === false) {
        const hint = data.hand_roi_error || "Hand ROI dependency unavailable"
        setError(`Hand ROI inactive: ${hint}`)
      }
    } else {
      const hint = data.model_error || "Model not loaded"
      setStatusChip(ui.modelStatus, "Model: not loaded", "status-warn")
      setError(hint)
    }
  } catch (err) {
    setStatusChip(ui.modelStatus, "Model: backend offline", "status-bad")
    setError(err.message)
  }
}

function classLookup() {
  const lookup = new Map()
  state.modelClasses.forEach((label) => {
    lookup.set(String(label).toUpperCase(), String(label))
  })
  return lookup
}

function spaceTokenForSequence(lookup) {
  if (lookup.has("SPACE")) {
    return lookup.get("SPACE")
  }
  return "<SPACE>"
}

function tokenizeTextToSigns(text) {
  const lookup = classLookup()
  const spaceToken = spaceTokenForSequence(lookup)
  const tokens = []
  const words = (text || "")
    .toUpperCase()
    .replace(/[^A-Z0-9_\s]/g, " ")
    .trim()
    .split(/\s+/)
    .filter(Boolean)

  for (let i = 0; i < words.length; i += 1) {
    const word = words[i]
    const directWord = lookup.get(word)
    if (directWord) {
      tokens.push(directWord)
    } else {
      for (const ch of word) {
        if (!/[A-Z0-9]/.test(ch)) {
          continue
        }
        tokens.push(lookup.get(ch) || ch)
      }
    }
    if (i < words.length - 1) {
      tokens.push(spaceToken)
    }
  }

  return tokens
}

function isPauseToken(token) {
  return String(token).toUpperCase() === "<SPACE>"
}

function tokenDisplayLabel(token) {
  return isPauseToken(token) ? "SPACE" : String(token)
}

function signSampleUrl(token) {
  const base = state.backendUrl.endsWith("/") ? state.backendUrl.slice(0, -1) : state.backendUrl
  return `${base}/sign-sample/${encodeURIComponent(String(token))}`
}

function setSignPreview(token, statusText) {
  ui.signPreviewStatus.textContent = statusText

  if (!token) {
    ui.signPreviewLabel.textContent = "-"
    ui.signPreviewImage.style.display = "none"
    ui.signPreviewFallback.style.display = "grid"
    ui.signPreviewFallback.textContent = "No preview"
    return
  }

  ui.signPreviewLabel.textContent = tokenDisplayLabel(token)
  if (isPauseToken(token)) {
    ui.signPreviewImage.style.display = "none"
    ui.signPreviewFallback.style.display = "grid"
    ui.signPreviewFallback.textContent = "Pause"
    return
  }

  ui.signPreviewImage.style.display = "none"
  ui.signPreviewFallback.style.display = "grid"
  ui.signPreviewFallback.textContent = "Loading..."
  ui.signPreviewImage.onerror = () => {
    ui.signPreviewImage.style.display = "none"
    ui.signPreviewFallback.style.display = "grid"
    ui.signPreviewFallback.textContent = `No sample for ${tokenDisplayLabel(token)}`
  }
  ui.signPreviewImage.onload = () => {
    ui.signPreviewFallback.style.display = "none"
    ui.signPreviewImage.style.display = "block"
  }
  ui.signPreviewImage.src = signSampleUrl(token)
}

function renderSignTimeline() {
  ui.signTokenTimeline.innerHTML = ""
  if (!state.signSeqTokens.length) {
    return
  }

  state.signSeqTokens.forEach((token, index) => {
    const chip = document.createElement("span")
    chip.className = "sign-token-chip"
    if (index < state.signSeqIndex) {
      chip.classList.add("done")
    }
    if (index === state.signSeqIndex) {
      chip.classList.add("active")
    }
    chip.textContent = tokenDisplayLabel(token)
    ui.signTokenTimeline.appendChild(chip)
  })
}

function syncSignPlayerButtons() {
  const hasTokens = state.signSeqTokens.length > 0
  ui.playSignSeqBtn.disabled = !hasTokens || state.signSeqPlaying
  ui.stopSignSeqBtn.disabled = !state.signSeqPlaying
}

function ensureTtsEnabledForTextToSign() {
  if (state.ttsEnabled) {
    return
  }
  state.ttsEnabled = true
  ui.ttsToggleBtn.textContent = "TTS: On"
}

function speakPhrase(phrase, options = {}) {
  if (!state.ttsEnabled) {
    return
  }
  if (!window.speechSynthesis) {
    setError("Speech synthesis is not supported in this browser")
    return
  }

  const text = String(phrase || "").trim()
  if (!text) {
    return
  }

  const { cancelCurrent = true, rate = 1.0, pitch = 1.0, lang = "en-IN" } = options
  if (cancelCurrent) {
    window.speechSynthesis.cancel()
  }

  const utterance = new SpeechSynthesisUtterance(text)
  utterance.lang = lang
  utterance.rate = rate
  utterance.pitch = pitch
  window.speechSynthesis.speak(utterance)
}

function speakTextToSignInput() {
  const typed = String(ui.textToSignInput.value || "").trim()
  if (!typed) {
    return
  }
  speakPhrase(typed, { cancelCurrent: true, rate: 0.96, pitch: 1.0, lang: "en-IN" })
}

function stopSignSequence(resetIndex = false) {
  if (state.signSeqTimer) {
    clearTimeout(state.signSeqTimer)
    state.signSeqTimer = null
  }
  state.signSeqPlaying = false
  if (resetIndex) {
    state.signSeqIndex = -1
    setSignPreview(null, "Idle")
  } else if (state.signSeqIndex >= state.signSeqTokens.length) {
    setSignPreview(null, "Completed")
  } else {
    setSignPreview(state.signSeqTokens[state.signSeqIndex] || null, "Paused")
  }
  renderSignTimeline()
  syncSignPlayerButtons()
}

function generateSignSequence() {
  stopSignSequence(true)
  const text = ui.textToSignInput.value || ""
  const tokens = tokenizeTextToSigns(text)
  state.signSeqTokens = tokens
  state.signSeqIndex = -1
  renderSignTimeline()
  syncSignPlayerButtons()

  if (!tokens.length) {
    setSignPreview(null, "No valid tokens")
    return
  }

  setSignPreview(tokens[0], `Ready (${tokens.length} tokens)`)
}

function stepSignSequence() {
  if (!state.signSeqPlaying) {
    return
  }

  state.signSeqIndex += 1
  if (state.signSeqIndex >= state.signSeqTokens.length) {
    stopSignSequence(false)
    return
  }

  const token = state.signSeqTokens[state.signSeqIndex]
  setSignPreview(token, `Playing ${state.signSeqIndex + 1}/${state.signSeqTokens.length}`)
  renderSignTimeline()
  state.signSeqTimer = setTimeout(stepSignSequence, state.signSeqFrameMs)
}

function playSignSequence() {
  if (!state.signSeqTokens.length) {
    generateSignSequence()
  }
  if (!state.signSeqTokens.length) {
    return
  }

  if (state.signSeqPlaying) {
    return
  }

  if (state.signSeqIndex >= state.signSeqTokens.length - 1) {
    state.signSeqIndex = -1
  }
  state.signSeqPlaying = true
  syncSignPlayerButtons()
  ensureTtsEnabledForTextToSign()
  speakTextToSignInput()
  stepSignSequence()
}

function useAssembledText() {
  const text = ui.committedText.textContent === "(empty)" ? "" : ui.committedText.textContent
  ui.textToSignInput.value = text
  generateSignSequence()
}

function renderTopk(topk = []) {
  ui.topkList.innerHTML = ""
  if (!topk.length) {
    const empty = document.createElement("li")
    empty.textContent = "No predictions"
    ui.topkList.appendChild(empty)
    return
  }

  topk.forEach((item) => {
    const li = document.createElement("li")
    li.textContent = `${item.label}: ${Number(item.conf).toFixed(3)}`
    ui.topkList.appendChild(li)
  })
}

function renderBuffer(bufferState = []) {
  ui.bufferList.innerHTML = ""
  if (!bufferState.length) {
    const empty = document.createElement("li")
    empty.textContent = "Buffer empty"
    ui.bufferList.appendChild(empty)
    return
  }

  bufferState.forEach((item) => {
    const li = document.createElement("li")
    li.textContent = `${item.label} (${Number(item.conf).toFixed(2)})`
    ui.bufferList.appendChild(li)
  })
}

function speakOnCommit(message) {
  if (!state.ttsEnabled || !window.speechSynthesis || !message.token_committed) {
    return
  }

  const token = message.committed_token
  let phrase = ""

  if (token === " ") {
    const words = (message.committed_text || "").trim().split(/\s+/)
    phrase = words[words.length - 1] || ""
  } else if (token === "<DELETE>") {
    phrase = "Deleted"
  } else if (typeof token === "string") {
    phrase = token
  }

  if (!phrase) {
    return
  }
  speakPhrase(phrase, { cancelCurrent: true, rate: 1.0, pitch: 1.0, lang: "en-IN" })
}

function flashCommitCue() {
  ui.commitFlashTarget.classList.remove("commit-flash")
  window.requestAnimationFrame(() => {
    ui.commitFlashTarget.classList.add("commit-flash")
  })
}

function humanLabel(label) {
  if (!label || label === state.unknownLabel) {
    return "No sign detected"
  }
  return label
}

function updatePredictionUI(message) {
  const rawConf = Number(message.raw_conf || 0)
  const rawLabel = rawConf >= state.minConfidence ? message.raw_pred : state.unknownLabel
  ui.rawSign.textContent = humanLabel(rawLabel)
  ui.stableSign.textContent = humanLabel(message.stable_pred)
  ui.stableSignPill.textContent = humanLabel(message.stable_pred)
  ui.confidence.textContent = rawConf.toFixed(3)
  ui.modelFps.textContent = Number(message.model_fps_estimate || 0).toFixed(1)
  ui.latency.textContent = `${Number(message.latency_ms || 0).toFixed(1)} ms`

  if (typeof message.network_latency_ms === "number") {
    ui.networkLatency.textContent = `${Number(message.network_latency_ms).toFixed(0)} ms`
  }

  const text = message.committed_text || ""
  ui.committedText.textContent = text.length ? text : "(empty)"

  if (message.token_committed) {
    flashCommitCue()
  }

  speakOnCommit(message)
  renderTopk(message.topk || [])
  renderBuffer(message.buffer_state || [])

  if (message.model_error) {
    setError(message.model_error)
  }
}

function closeSocket() {
  resetFrameInFlight(true)
  if (state.socket) {
    state.socket.onclose = null
    state.socket.close()
    state.socket = null
  }
}

function scheduleReconnect() {
  resetFrameInFlight(true)
  if (!state.cameraRunning || state.reconnectTimer) {
    return
  }

  state.reconnectAttempts += 1
  const delay = Math.min(1000 * 2 ** (state.reconnectAttempts - 1), 8000)
  setStatusChip(ui.socketStatus, `Connection: reconnecting in ${Math.round(delay / 1000)}s`, "status-warn")

  state.reconnectTimer = setTimeout(() => {
    state.reconnectTimer = null
    connectSocket()
  }, delay)
}

function connectSocket() {
  resetFrameInFlight(true)
  if (state.socket && (state.socket.readyState === WebSocket.OPEN || state.socket.readyState === WebSocket.CONNECTING)) {
    return
  }

  try {
    const ws = new WebSocket(asWsUrl(state.backendUrl))
    state.socket = ws
    setStatusChip(ui.socketStatus, "Connection: connecting", "status-warn")

    ws.onopen = () => {
      resetFrameInFlight(true)
      state.reconnectAttempts = 0
      setStatusChip(ui.socketStatus, "Connection: connected", "status-good")
      setError("")
    }

    ws.onmessage = (event) => {
      let msg
      try {
        msg = JSON.parse(event.data)
      } catch (err) {
        setError(`Invalid server JSON: ${err.message}`)
        return
      }

      if (msg.type === "prediction") {
        resetFrameInFlight()
        updatePredictionUI(msg)
        return
      }

      if (msg.type === "control") {
        ui.committedText.textContent = msg.committed_text || "(empty)"
        if (!msg.ok) {
          setError(msg.error || "Control action failed")
        }
        return
      }

      if (msg.type === "status") {
        if (typeof msg.model_loaded === "boolean") {
          if (msg.model_loaded) {
            setStatusChip(ui.modelStatus, "Model: loaded", "status-good")
          } else {
            setStatusChip(ui.modelStatus, "Model: not loaded", "status-warn")
          }
        }
        if (msg.model_error) {
          setError(msg.model_error)
        }
        return
      }

      if (msg.type === "error") {
        resetFrameInFlight(true)
        setError(msg.error || "Server error")
      }
    }

    ws.onerror = () => {
      resetFrameInFlight(true)
      setStatusChip(ui.socketStatus, "Connection: error", "status-bad")
      setError("WebSocket connection error")
    }

    ws.onclose = () => {
      resetFrameInFlight(true)
      state.socket = null
      if (!state.cameraRunning) {
        setStatusChip(ui.socketStatus, "Connection: disconnected", "status-warn")
        return
      }
      scheduleReconnect()
    }
  } catch (err) {
    setStatusChip(ui.socketStatus, "Connection: failed", "status-bad")
    setError(err.message)
  }
}

function drawSquareFrame() {
  const video = ui.webcam
  if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    return null
  }

  const width = video.videoWidth
  const height = video.videoHeight
  if (!width || !height) {
    return null
  }

  const side = Math.min(width, height)
  const sx = Math.floor((width - side) / 2)
  const sy = Math.floor((height - side) / 2)

  sendCanvas.width = state.frameSize
  sendCanvas.height = state.frameSize

  sendCtx.drawImage(video, sx, sy, side, side, 0, 0, state.frameSize, state.frameSize)
  return sendCanvas.toDataURL("image/jpeg", 0.72)
}

function sendFrame() {
  if (!state.cameraRunning || !state.socket) {
    return
  }

  if (state.frameInFlight) {
    return
  }

  const readyState = state.socket.readyState
  if (readyState === WebSocket.CONNECTING || readyState === WebSocket.CLOSING || readyState === WebSocket.CLOSED) {
    return
  }

  if (readyState !== WebSocket.OPEN) {
    return
  }

  const image = drawSquareFrame()
  if (!image) {
    return
  }

  const payload = {
    type: "frame",
    image,
    ts: Date.now(),
  }

  try {
    state.socket.send(JSON.stringify(payload))
    markFrameInFlight()
  } catch (err) {
    resetFrameInFlight(true)
    setError(`Frame send failed: ${err.message}`)
  }
}

function startSendLoop() {
  if (state.sendTimer) {
    clearInterval(state.sendTimer)
  }

  const intervalMs = Math.round(1000 / state.sendFps)
  state.processingIdleDelayMs = Math.max(350, intervalMs * 2)
  state.sendTimer = setInterval(sendFrame, intervalMs)
}

function stopSendLoop() {
  if (state.sendTimer) {
    clearInterval(state.sendTimer)
    state.sendTimer = null
  }
}

async function startCamera() {
  if (state.cameraRunning) {
    return
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: "user",
        width: { ideal: 960 },
        height: { ideal: 720 },
      },
    })

    state.mediaStream = stream
    ui.webcam.srcObject = stream
    await ui.webcam.play()

    state.cameraRunning = true
    ui.startCameraBtn.disabled = true
    ui.stopCameraBtn.disabled = false
    setStatusChip(ui.cameraStatus, "Camera: running", "status-good")

    connectSocket()
    startSendLoop()
  } catch (err) {
    setStatusChip(ui.cameraStatus, "Camera: denied/unavailable", "status-bad")
    setError(`Camera error: ${err.message}`)
  }
}

function stopCamera() {
  state.cameraRunning = false
  resetFrameInFlight(true)

  if (state.reconnectTimer) {
    clearTimeout(state.reconnectTimer)
    state.reconnectTimer = null
  }

  stopSendLoop()

  if (state.mediaStream) {
    state.mediaStream.getTracks().forEach((track) => track.stop())
    state.mediaStream = null
  }

  ui.webcam.srcObject = null
  ui.startCameraBtn.disabled = false
  ui.stopCameraBtn.disabled = true
  setStatusChip(ui.cameraStatus, "Camera: stopped", "status-warn")

  closeSocket()
  setStatusChip(ui.socketStatus, "Connection: disconnected", "status-warn")
}

function sendControl(action) {
  if (!state.socket || state.socket.readyState !== WebSocket.OPEN) {
    setError("WebSocket not connected")
    return
  }
  state.socket.send(JSON.stringify({ type: "control", action }))
}

function toggleTts() {
  state.ttsEnabled = !state.ttsEnabled
  ui.ttsToggleBtn.textContent = `TTS: ${state.ttsEnabled ? "On" : "Off"}`
  if (!state.ttsEnabled && window.speechSynthesis) {
    window.speechSynthesis.cancel()
  }
}

function toggleDebug() {
  state.debugVisible = !state.debugVisible
  ui.debugPanel.classList.toggle("hidden", !state.debugVisible)
}

function saveBackendUrl() {
  const value = ui.backendUrl.value.trim()
  if (!value) {
    return
  }

  try {
    new URL(value)
  } catch {
    setError("Invalid backend URL")
    return
  }

  state.backendUrl = value
  localStorage.setItem("isl_backend_url", value)
  refreshHealth()

  if (state.cameraRunning) {
    closeSocket()
    connectSocket()
  }
}

function bindEvents() {
  ui.startCameraBtn.addEventListener("click", startCamera)
  ui.stopCameraBtn.addEventListener("click", stopCamera)
  ui.clearTextBtn.addEventListener("click", () => sendControl("clear_text"))
  ui.backspaceBtn.addEventListener("click", () => sendControl("backspace"))
  ui.ttsToggleBtn.addEventListener("click", toggleTts)
  ui.debugToggleBtn.addEventListener("click", toggleDebug)
  ui.saveBackendBtn.addEventListener("click", saveBackendUrl)
  ui.useAssembledBtn.addEventListener("click", useAssembledText)
  ui.generateSignSeqBtn.addEventListener("click", generateSignSequence)
  ui.playSignSeqBtn.addEventListener("click", playSignSequence)
  ui.stopSignSeqBtn.addEventListener("click", () => stopSignSequence(false))
  ui.textToSignInput.addEventListener("input", () => {
    if (state.signSeqPlaying) {
      stopSignSequence(false)
    }
  })
}

function init() {
  loadBackendUrl()
  bindEvents()
  refreshHealth()
  window.setInterval(refreshHealth, 5000)
  setStatusChip(ui.cameraStatus, "Camera: idle", "status-warn")
  setStatusChip(ui.socketStatus, "Connection: disconnected", "status-warn")
  setStatusChip(ui.modelStatus, "Model: checking...", "status-warn")
  resetFrameInFlight(true)
  setSignPreview(null, "Idle")
  syncSignPlayerButtons()
}

init()
