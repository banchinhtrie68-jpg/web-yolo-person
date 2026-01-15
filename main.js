const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const countEl = document.getElementById("count");

let session;
const SIZE = 640;

async function start() {
    session = await ort.InferenceSession.create("yolov8n.onnx");

    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;

    video.onloadeddata = () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        detect();
    };
}

async function detect() {
    ctx.drawImage(video, 0, 0, SIZE, SIZE);
    const img = ctx.getImageData(0, 0, SIZE, SIZE);

    const input = preprocess(img);
    const output = await session.run({ images: input });

    const persons = drawBoxes(output.output0.data);
    countEl.innerText = persons;

    requestAnimationFrame(detect);
}

function preprocess(img) {
    const data = new Float32Array(3 * SIZE * SIZE);
    for (let i = 0; i < img.data.length; i += 4) {
        const j = (i / 4) * 3;
        data[j] = img.data[i] / 255;
        data[j + 1] = img.data[i + 1] / 255;
        data[j + 2] = img.data[i + 2] / 255;
    }
    return new ort.Tensor("float32", data, [1, 3, SIZE, SIZE]);
}

function drawBoxes(out) {
    let count = 0;
    ctx.strokeStyle = "#22c55e";
    ctx.lineWidth = 2;

    for (let i = 0; i < out.length; i += 85) {
        const score = out[i + 4];
        const cls = out[i + 5];

        if (score > 0.5 && cls === 0) {
            count++;
            const x = out[i] * canvas.width;
            const y = out[i + 1] * canvas.height;
            const w = out[i + 2] * canvas.width;
            const h = out[i + 3] * canvas.height;

            ctx.strokeRect(x - w / 2, y - h / 2, w, h);
        }
    }
    return count;
}