const webcamElement = document.getElementById('webcam');
const webcamElement2 = document.getElementById('webcam2');
const classifier = knnClassifier.create();

let net;

async function app() {
	console.log('Loading mobilenet..');

	// Load the model.
	net = await mobilenet.load();
	console.log('Successfully loaded model');

	// Make a prediction through the model on our image.
	const imgEl = document.getElementById('img');
	const result3 = await net.classify(imgEl);
	console.log(result3);

	document.getElementById('console3').innerText = `
      prediction: ${result3[0].className}\n
      probability: ${result3[0].probability}
    `;
	// Create an object from Tensorflow.js data API which could capture image
	// from the web camera as Tensor.
	const webcam = await tf.data.webcam(webcamElement);
	const webcam2 = await tf.data.webcam(webcamElement2);
	// Reads an image from the webcam and associates it with a specific class
	// index.
	const addExample = async (classId) => {
		// Capture an image from the web camera.
		const img = await webcam.capture();

		// Get the intermediate activation of MobileNet 'conv_preds' and pass that
		// to the KNN classifier.
		const activation = net.infer(img, true);

		// Pass the intermediate activation to the classifier.
		classifier.addExample(activation, classId);

		// Dispose the tensor to release the memory.
		img.dispose();
	};

	// When clicking a button, add an example for that class.
	document
		.getElementById('class-a')
		.addEventListener('click', () => addExample(0));
	document
		.getElementById('class-b')
		.addEventListener('click', () => addExample(1));
	document
		.getElementById('class-c')
		.addEventListener('click', () => addExample(2));

	while (true) {
		const img2 = await webcam2.capture();
		const result2 = await net.classify(img2);

		document.getElementById('console2').innerText = `
      prediction: ${result2[0].className}\n
      probability: ${result2[0].probability}
    `;
		// Dispose the tensor to release the memory.
		img2.dispose();

		if (classifier.getNumClasses() > 0) {
			const img = await webcam.capture();

			// Get the activation from mobilenet from the webcam.
			const activation = net.infer(img, 'conv_preds');
			// Get the most likely class and confidence from the classifier module.
			const result = await classifier.predictClass(activation);

			const classes = ['A', 'B', 'C'];
			document.getElementById('console').innerText = `
        prediction: ${classes[result.label]}\n
        probability: ${result.confidences[result.label]}
      `;

			// Dispose the tensor to release the memory.
			img.dispose();
		}

		await tf.nextFrame();
	}
}
app();
