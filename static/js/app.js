const drawCanvas = document.getElementById("drawCanvas");
const hiddenCanvas = document.getElementById("hiddenCanvas");
const drawCtx = drawCanvas.getContext("2d");
const hiddenCtx = hiddenCanvas.getContext("2d");

const predictBtn = document.getElementById("predictBtn");
const clearBtn = document.getElementById("clearBtn");
const resultCard = document.getElementById("resultCard");
const placeholder = document.getElementById("placeholder");
const predictedDigit = document.getElementById("predictedDigit");
const confidenceEl = document.getElementById("confidence");
const probBars = document.getElementById("probBars");

let isDrawing = false;

function initCanvas() {
    drawCtx.fillStyle = "#ffffff";
    drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
    drawCtx.strokeStyle = "#000000";
    drawCtx.lineWidth = 18;
    drawCtx.lineCap = "round";
    drawCtx.lineJoin = "round";
}

function getPos(event) {
    const rect = drawCanvas.getBoundingClientRect();
    const scaleX = drawCanvas.width / rect.width;
    const scaleY = drawCanvas.height / rect.height;
    const clientX = event.touches ? event.touches[0].clientX : event.clientX;
    const clientY = event.touches ? event.touches[0].clientY : event.clientY;
    return {
        x: (clientX - rect.left) * scaleX,
        y: (clientY - rect.top) * scaleY,
    };
}

function startDraw(event) {
    event.preventDefault();
    isDrawing = true;
    const { x, y } = getPos(event);
    drawCtx.beginPath();
    drawCtx.moveTo(x, y);
}

function draw(event) {
    if (!isDrawing) return;
    event.preventDefault();
    const { x, y } = getPos(event);
    drawCtx.lineTo(x, y);
    drawCtx.stroke();
}

function stopDraw() {
    isDrawing = false;
}

function clearCanvas() {
    initCanvas();
    resultCard.classList.add("hidden");
    placeholder.classList.remove("hidden");
    predictedDigit.textContent = "—";
    confidenceEl.textContent = "Confidence: —";
    probBars.innerHTML = "";
}

function getPixelData() {
    hiddenCtx.drawImage(drawCanvas, 0, 0, 28, 28);
    const imageData = hiddenCtx.getImageData(0, 0, 28, 28);
    const pixels = [];

    for (let i = 0; i < imageData.data.length; i += 4) {
        const r = imageData.data[i];
        const g = imageData.data[i + 1];
        const b = imageData.data[i + 2];
        pixels.push(Math.round((r + g + b) / 3));
    }

    return pixels;
}

function renderProbabilities(probabilities, topDigit) {
    probBars.innerHTML = "";
    const sorted = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);

    sorted.forEach(([digit, pct]) => {
        const row = document.createElement("div");
        row.className = "prob-row" + (digit === String(topDigit) ? " top" : "");

        row.innerHTML = `
            <span class="digit">${digit}</span>
            <div class="prob-track"><div class="prob-fill" style="width: ${pct}%"></div></div>
            <span class="pct">${pct}%</span>
        `;
        probBars.appendChild(row);
    });
}

async function predict() {
    predictBtn.classList.add("loading");
    predictBtn.textContent = "Predicting...";

    try {
        const pixels = getPixelData();
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ pixels }),
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Prediction failed");
        }

        placeholder.classList.add("hidden");
        resultCard.classList.remove("hidden");
        predictedDigit.textContent = data.digit;
        confidenceEl.textContent = `Confidence: ${data.confidence}%`;
        renderProbabilities(data.probabilities, data.digit);
    } catch (err) {
        alert(err.message);
    } finally {
        predictBtn.classList.remove("loading");
        predictBtn.textContent = "Predict";
    }
}

drawCanvas.addEventListener("mousedown", startDraw);
drawCanvas.addEventListener("mousemove", draw);
drawCanvas.addEventListener("mouseup", stopDraw);
drawCanvas.addEventListener("mouseleave", stopDraw);

drawCanvas.addEventListener("touchstart", startDraw, { passive: false });
drawCanvas.addEventListener("touchmove", draw, { passive: false });
drawCanvas.addEventListener("touchend", stopDraw);

predictBtn.addEventListener("click", predict);
clearBtn.addEventListener("click", clearCanvas);

initCanvas();
