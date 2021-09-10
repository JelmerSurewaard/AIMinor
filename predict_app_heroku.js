/*
This node application starts an http server and loads the pre-trained Keras/Tensorflow model, from 
hands-on 3 'univariate linear regression using a keras/tensorflow NN'. By passing a city size this
application returns the expected profit, using  the Keras/Tensorflow neural network.

To run this application: command prompt> node my_predict_app.js
To use this application: open browser with http://localhost:8081/?citysize=20
The browser should show an expected profit of 192580.92880249023 dollars
*/

const http = require("http");
const url = require('url');
const tf = require('@tensorflow/tfjs');
const tfn = require("@tensorflow/tfjs-node");

const hostname = '0.0.0.0';
const port = process.env.PORT || 8081;

http.createServer((request, response) => {
    const query = url.parse(request.url, true).query;
    response.writeHead(200, {'Content-Type': 'text/plain'});
    if (query.citysize) predictProfit(query.citysize)
        .then(res => response.end('The expected profit for a city size of ' + query.citysize + '0000 is: ' + res * 10000 + ' dollars.')).catch(err => console.log('error: ', err));
    else response.end(`No city size provided, try: http://${hostname}:${port}/?citysize=20`)
}).listen(port, hostname, () => {
    console.log(`Server running at http://${hostname}:${port}, try: http://${hostname}:${port}/?citysize=20`);
});

function featureNormalize(citysize) {
    trainingSetMean = 8.159799999999999; // value taken from hands-on 3 'univariate linear regression using a keras/tensorflow NN'
    trainingSetStd = 3.849883995922715; // value taken from hands-on 3 'univariate linear regression using a keras/tensorflow NN'
    return (citysize - trainingSetMean) / trainingSetStd;
}

var my_regression_model;
async function predictProfit(citysize) {
    // only load model once to save resources
    if(!my_regression_model) {
        console.log('load pre-trained regression model')
        const handler = tfn.io.fileSystem('./univariateKerasRegression_tfjs_model/model.json'); // read trained Keras model from disk
        my_regression_model = await tf.loadLayersModel(handler); // load model into tensorflow.js
    }
    console.log('input: ' + citysize);
    citysize = featureNormalize(citysize); // don't forget to feature-normalize!!
    let result = my_regression_model.predict(tf.tensor1d([parseFloat(citysize)])).arraySync(); // do prediction
    console.log('result: ' + result);
    return result;
}

