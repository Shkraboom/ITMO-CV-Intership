const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const toggleButton = document.getElementById('toggleButton');

let lastTimestamp = performance.now();
let frameCount = 0;
let lastFPSUpdate = performance.now();
let fps = 0;
let useWebcam = true;

async function setupCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    return new Promise(resolve => video.onloadedmetadata = () => resolve(video));
}

async function loadVideo() {
    return new Promise((resolve) => {
        video.src = "data/6387-191695740_small.mp4";
        video.loop = true;
        video.muted = true;
        video.autoplay = true;
        video.onloadeddata = () => resolve(video);
    });
}

async function loadModel() {
    return await tf.loadGraphModel('yolo11n_web_model/model.json');
}

async function detect(model) {
    if (video.readyState < 2) {
        requestAnimationFrame(() => detect(model));
        return;
    }
    
    const startTime = performance.now();
    const tensor = tf.browser.fromPixels(video)
        .resizeBilinear([640, 640])
        .expandDims(0)
        .toFloat().div(255.0);

    const predictions = await model.executeAsync(tensor);
    const inferenceTime = performance.now() - startTime;

    frameCount++;
    const now = performance.now();
    if (now - lastFPSUpdate >= 1000) {
        fps = frameCount;
        frameCount = 0;
        lastFPSUpdate = now;
    }

    console.log(`Inference time: ${inferenceTime.toFixed(2)} ms | FPS: ${fps}`);

    const predArray = predictions.arraySync()[0];
    if (!predArray || predArray.length !== 84) {
        console.error("Unexpected predictions structure:", predArray);
        return;
    }

    const boxes = [predArray[0], predArray[1], predArray[2], predArray[3]];
    const scores = predArray[4];
    const classes = predArray.slice(5, 84).map(arr => arr.indexOf(Math.max(...arr)));

    drawPredictions(boxes, scores, classes, inferenceTime, fps);
    requestAnimationFrame(() => detect(model));
}

function drawPredictions(boxes, scores, classes, inferenceTime, fps) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    ctx.fillStyle = "rgba(0, 0, 0, 0.5)";
    ctx.fillRect(10, 10, 150, 40);
    ctx.fillStyle = "white";
    ctx.font = "16px Arial";
    ctx.fillText(`FPS: ${fps}`, 20, 30);
    ctx.fillText(`Time: ${inferenceTime.toFixed(2)}ms`, 20, 50);

    const scaleX = canvas.width / 640;
    const scaleY = canvas.height / 640;

    for (let i = 0; i < boxes[0].length; i++) {
        let cx = boxes[0][i] * scaleX;
        let cy = boxes[1][i] * scaleY;
        let width = boxes[2][i] * scaleX;
        let height = boxes[3][i] * scaleY;

        let x1 = cx - width / 2;
        let y1 = cy - height / 2;

        let score = scores[i];

        if (score > 0.5) {
            ctx.strokeStyle = 'green';
            ctx.lineWidth = 2;
            ctx.strokeRect(x1, y1, width, height);
            ctx.fillStyle = 'green';
            ctx.fillText(`Person: ${(score * 100).toFixed(1)}%`, x1, y1 > 10 ? y1 - 5 : 10);
        }
    }
}

async function startDetection() {
    const model = await loadModel();
    detect(model);
}

async function switchSource() {
    useWebcam = !useWebcam;
    
    if (useWebcam) {
        await setupCamera();
    } else {
        await loadVideo();
    }
    
    startDetection();
}

toggleButton.addEventListener('click', switchSource);

async function main() {
    await setupCamera();
    startDetection();
}

main();
