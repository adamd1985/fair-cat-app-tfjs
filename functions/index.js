/**
 * Cloud function
 * @adamd1985
 * @see https://firebase.google.com/docs/functions
 */

const { onRequest } = require("firebase-functions/v2/https");
const logger = require("firebase-functions/logger");
const admin = require('firebase-admin');
const tf = require('@tensorflow/tfjs');
const tfn = require('@tensorflow/tfjs-node');
const fs = require('fs');
let DATASETS_METADATA = {};
let MODEL = null;
const FUNCTION_MODEL_PATH = './models';
const BRANDS = ['Unbranded', 'Whiskers and Paws', 'Royal Feline', 'Meowarf'];
const STORES = ['Fresh Pet', 'Expensive Cats', 'Overpriced Pets', 'Jungle of Money', 'Mom & Pop Petshop'];

/**
 * OHE helper for categories.
 * 
 * @param {*} labels 
 * @returns 
 */
function oneHotEncode(labels, labelConstants, featureName) {
    let encodedTensor = null;
    const encodedLabels = labels.map(label => labelConstants.indexOf(label));

    if (!DATASETS_METADATA[`${featureName}_OHE`]) {
        DATASETS_METADATA[`${featureName}_OHE`] = encodedLabels;
    }

    encodedTensor = tf.oneHot(tf.tensor1d(encodedLabels, 'int32'), labelConstants.length);

    return encodedTensor;
}


/**
 * Normalize to their relative weights.
 * 
 * @see DATASETS_METADATA
 * @param feature A tensor with the features to be normalized.
 * @param featureName The feature name
 * @param min Minimum value to be used in the normalization. Defaults to NULL. If Null it will save the value fromt the given feature range using metadata.
 * @param max Maximum value to be used in the normalization. Defaults to NULL. If Null it will save the value fromt the given feature range using metadata.
 * @param metaData Meta data to save features.
 * @returns {Array[*]} The normalized range.
 */
function normalizeFeature(feature, featureName, metaData = DATASETS_METADATA) {
    const min = tf.tensor1d([metaData[`${featureName}_norm`].min]);
    const max = tf.tensor1d([metaData[`${featureName}_norm`].max]);


    const normalizedFeature = tf.div(tf.sub(feature, min), tf.sub(max, min));

    return normalizedFeature;
}

/**
 * Loads meta data and model.
 * 
 * Once loaded, warm up model with sample prediciton.
 */
function loadModel() {
    fs.readFile(`${FUNCTION_MODEL_PATH}/meta.json`, (err, data) => {
        if (err) throw err;

        logger.info(`Model metadata loaded ${data}`);

        DATASETS_METADATA = JSON.parse(data);

        const brand = oneHotEncode([BRANDS[1]], BRANDS, 'brand');
        const weightInGrams = tf.tensor1d([5000]);
        const weight = normalizeFeature(weightInGrams, 'weight');

        tf.loadLayersModel(tfn.io.fileSystem(`${FUNCTION_MODEL_PATH}/model.json`))
            .then((loadedModel) => {
                logger.info(`Model loaded ${loadedModel}, predicting sample: `);

                const x = tf.tensor2d(
                    tf.concat([
                        tf.tensor2d(weight.arraySync(), [weight.arraySync().length, 1]),
                        tf.tensor2d(brand.arraySync())], 1)
                        .arraySync());
                MODEL = loadedModel;

                return MODEL.predict(x);
            }).then((prediction) => {
                logger.info(`Predicted: '$${prediction}' for a brand: '${BRANDS[1]}' and weight: '${weightInGrams}g'`);
            });
    });

}

loadModel();

admin.initializeApp();
const database = admin.database();

/**
 * POST only, predicts the price of the catfood item.
 */
exports.catFoodPredictor = onRequest(async (req, res) => {
    if (req.method !== 'POST') {
        return res.status(400).json({ error: 'Invalid request method. Only POST requests are allowed.' });
    }

    const data = req.body;
    logger.info(`Received this: ${JSON.stringify(data)}`);


    await database.ref('telemetry').push({
        data: JSON.stringify(data),
        timestamp: Date.now(),
    });

    logger.info(`Received this: ${data.brand} and ${data.weight}`);

    const brand = oneHotEncode([data.brand], BRANDS, 'brand');
    const weightInGrams = tf.tensor1d([data.weight]);
    const weight = normalizeFeature(weightInGrams, 'weight');

    const x = tf.tensor2d(
        tf.concat([
            tf.tensor2d(weight.arraySync(), [weight.arraySync().length, 1]),
            tf.tensor2d(brand.arraySync())], 1)
            .arraySync());

    try {
        const prediciton = MODEL.predict(x).arraySync()[0];
        res.status(200).json({ prediciton: prediciton });

        logger.info(`Predicted this: ${JSON.stringify(prediciton)}`);
    }
    catch (err) {
        console.error('Error adding data:', error);
        res.status(500).json({ error: 'Something went wrong. Please try again later.' });
    }
});

