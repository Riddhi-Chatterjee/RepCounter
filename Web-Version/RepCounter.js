const video5 = document.getElementsByClassName('input_video5')[0];
const out5 = document.getElementsByClassName('output5')[0];
const controlsElement5 = document.getElementsByClassName('control5')[0];
const canvasCtx5 = out5.getContext('2d');

const fpsControl = new FPS();

const spinner = document.querySelector('.loading');
spinner.ontransitionend = () => {
    spinner.style.display = 'none';
};

function zColor(data) {
    const z = clamp(data.from.z + 0.5, 0, 1);
    return `rgba(0, ${255 * z}, ${255 * (1 - z)}, 1)`;
}

data = []
init_dir = "None"
control = false

function onResultsPose(results) {
    var pl = results.poseLandmarks
    //console.log(pl[0])
    var plList = []
    var i;
    for(i=0;i<33;i++)
    {
        var kp = []
        kp.push(i)
        kp.push(pl[i].x)
        kp.push(pl[i].y)
        kp.push(pl[i].z)
        kp.push(pl[i].visibility)
        plList.push(kp)
    }

    //console.log(plList)

    const jsComputeRepCount = pyodideGlobals.get('computeRepCount');
    // console.log(data)
    var info = jsComputeRepCount(plList, 30, data, init_dir, control) //Dummy frame rate being passed. Frame rate not needed for angle based features
    data = info.split("]")[0].split("[")[2].split(", ")
    var repCount = info.split("]")[1].split(", ")[1]
    init_dir = info.split("]")[1].split(", ")[2].split("'")[1]
    control = info.split("]")[1].split(", ")[3]

    // console.log(`info: ${info}`);
    // console.log(`data: ${data}`);
    // console.log(`init_dir: ${init_dir}`);
    // console.log(`control: ${control}`);
    console.log(`repCount: ${repCount}`);
    
    document.body.classList.add('loaded');
    fpsControl.tick();

    canvasCtx5.save();
    canvasCtx5.clearRect(0, 0, out5.width, out5.height);
    canvasCtx5.drawImage(
        results.image, 0, 0, out5.width, out5.height);
    drawConnectors(
        canvasCtx5, results.poseLandmarks, POSE_CONNECTIONS, {
            color: (data) => {
            const x0 = out5.width * data.from.x;
            const y0 = out5.height * data.from.y;
            const x1 = out5.width * data.to.x;
            const y1 = out5.height * data.to.y;

            const z0 = clamp(data.from.z + 0.5, 0, 1);
            const z1 = clamp(data.to.z + 0.5, 0, 1);

            const gradient = canvasCtx5.createLinearGradient(x0, y0, x1, y1);
            gradient.addColorStop(
                0, `rgba(0, ${255 * z0}, ${255 * (1 - z0)}, 1)`);
            gradient.addColorStop(
                1.0, `rgba(0, ${255 * z1}, ${255 * (1 - z1)}, 1)`);
            return gradient;
            }
    });
    drawLandmarks(
        canvasCtx5,
        Object.values(POSE_LANDMARKS_LEFT)
            .map(index => results.poseLandmarks[index]),
        {color: zColor, fillColor: '#FF0000'});
    drawLandmarks(
        canvasCtx5,
        Object.values(POSE_LANDMARKS_RIGHT)
            .map(index => results.poseLandmarks[index]),
        {color: zColor, fillColor: '#00FF00'});
    drawLandmarks(
        canvasCtx5,
        Object.values(POSE_LANDMARKS_NEUTRAL)
            .map(index => results.poseLandmarks[index]),
        {color: zColor, fillColor: '#AAAAAA'});
    canvasCtx5.restore();
}

const pose = new Pose({locateFile: (file) => {
    return `https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.2/${file}`;
}});
pose.onResults(onResultsPose);

const camera = new Camera(video5, {
    onFrame: async () => {
        await pose.send({image: video5});
    },
    width: 480,
    height: 480
});
camera.start();

new ControlPanel(controlsElement5, {
      selfieMode: true,
      upperBodyOnly: false,
      smoothLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
}).add([
    new StaticText({title: 'MediaPipe Pose'}),
    fpsControl,
    new Toggle({title: 'Selfie Mode', field: 'selfieMode'}),
    new Toggle({title: 'Upper-body Only', field: 'upperBodyOnly'}),
    new Toggle({title: 'Smooth Landmarks', field: 'smoothLandmarks'}),
    new Slider({
        title: 'Min Detection Confidence',
        field: 'minDetectionConfidence',
        range: [0, 1],
        step: 0.01
    }),
    new Slider({
        title: 'Min Tracking Confidence',
        field: 'minTrackingConfidence',
        range: [0, 1],
        step: 0.01
    }),
]).on(options => {
    video5.classList.toggle('selfie', options.selfieMode);
    pose.setOptions(options);
});
