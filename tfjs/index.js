/**
 * Tensorflow JS Analysis and Model Building.
 * @adamd1985
 */

import * as tf from '@tensorflow/tfjs-node'
import { plot } from 'nodeplotlib';
import Plot from 'nodeplotlib';
import * as fs from 'fs';
const { tidy, tensor2d } = tf;

// Constants
const BRANDS = ['Unbranded', 'Whiskers and Paws', 'Royal Feline', 'Meowarf'];
const STORES = ['Fresh Pet', 'Expensive Cats', 'Overpriced Pets', 'Jungle of Money', 'Mom & Pop Petshop'];
const MAX_DS_X = 1500;
const EPOCHS = 15;
const DATASETS_METADATA = {};
const FUNCTION_MODEL_PATH = "./functions/models"
/**
 * Generates random cat food data, either as normal or uniform data.
 * 
 * @param numRows The size of the dataset in X
 * @returns 2darray of features.
 */
function generateData(numRows,
    wieghtRangeGrams = { min: 1000.0, max: 10000.0 },
    brands = BRANDS,
    stores = STORES) {

    const brandIndices = tf.randomUniform([numRows], 0, brands.length, 'int32');
    const brandLabels = brandIndices.arraySync().map(index => brands[index]);
    const locationIndices = tf.randomUniform([numRows], 0, stores.length, 'int32');
    const locationLabels = locationIndices.arraySync().map(index => stores[index]);

    const bestBeforeDates = tf.randomUniform([numRows], 0, 365 * 5, 'int32');
    const baseDate = new Date();
    const bestBeforeDatesFormatted = bestBeforeDates.arraySync().map(days => {
        const date = new Date(baseDate);
        date.setDate(baseDate.getDate() + days);
        return date.toISOString().split('T')[0];
    });

    // Generate price values based on weights (with minor variance)
    const weights = tf.randomUniform([numRows], wieghtRangeGrams.min, wieghtRangeGrams.max, 'float32');

    const pricesTemp = weights.div(120);
    const priceMean = tf.mean(pricesTemp).arraySync(); // Mean weight
    const priceStd = tf.moments(pricesTemp).variance.sqrt().arraySync();
    const priceNoise = tf.randomNormal([numRows], priceMean, priceStd, 'float32');
    let prices = tf.tensor1d(pricesTemp.add(priceMean).add(priceNoise).arraySync());

    // Apply logic and transform each number
    prices = tf.tensor1d(prices.dataSync().map((value, index) => {
        const brandLabel = brandLabels[index];
        let newPrice = value;
        switch (brandLabel) {
            case 'Unbranded':
                newPrice *= 0.82;
                break;

            case 'Royal Feline':
                newPrice *= 1.12;
                newPrice += 10;
                break;

            case 'Whiskers and Paws':
                newPrice *= 1.45;
                newPrice += 25;
                break;

            case 'Meowarf':
                newPrice *= 1.60;
                newPrice += 50;
                break;

            default:
                throw new Error(brandLabel);
        }
        return newPrice;
    }));


    const data = {
        weight: weights.arraySync(),
        brand: brandLabels,
        storeLocation: locationLabels,
        bestBeforeDate: bestBeforeDatesFormatted,
        priceUSD: prices.arraySync(),
    };

    return data;
};

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
 * Does some EDA on the given data.
 * 
 * @param {*} {
 *       weight: aray of floats,
 *       brand: array of label strings,
 *       storeLocation: array of label strings,
 *       bestBeforeDate: array of iso dates,
 *       priceUSD: aray of floats,
 *   }; 
 */
function dataEDA(data) {
    function _countUniqueLabels(labels) {
        return labels.reduce((counts, label) => {
            counts[label] = (counts[label] || 0) + 1;
            return counts;
        }, {});
    }

    const { weight, brand, storeLocation, bestBeforeDate, priceUSD } = data;

    // Summary statistics
    const weightMean = tf.mean(weight);
    const weightStd = tf.moments(weight).variance.sqrt().arraySync();
    const priceMean = tf.mean(priceUSD);
    const priceStd = tf.moments(priceUSD).variance.sqrt().arraySync();

    console.log('Weight Summary:');
    console.log(`Mean: ${weightMean.dataSync()[0].toFixed(2)}`);
    console.log(`Standard Deviation: ${weightStd}`);
    console.log('\nPrice Summary:');
    console.log(`Mean: ${priceMean.dataSync()[0].toFixed(2)}`);
    console.log(`Standard Deviation: ${priceStd}`);

    // Histogram of weights
    const weightData = [{ x: weight, type: 'histogram' }];
    const weightLayout = { title: 'Weight Distribution' };
    plot(weightData, weightLayout);

    // Scatter plot of weight vs. price
    let scatterData = [
        { x: weight, y: priceUSD, mode: 'markers', type: 'scatter' },
    ];
    let scatterLayout = { title: 'Weight vs. Price', xaxis: { title: 'Weight' }, yaxis: { title: 'Price' } };
    plot(scatterData, scatterLayout);

    scatterData = [
        { x: brand, y: priceUSD, mode: 'markers', type: 'scatter' },
    ];
    scatterLayout = { title: 'Brand vs. Price', xaxis: { title: 'Brand' }, yaxis: { title: 'Price' } };
    plot(scatterData, scatterLayout);

    scatterData = [
        { x: storeLocation, y: priceUSD, mode: 'markers', type: 'scatter' },
    ];
    scatterLayout = { title: 'Store vs. Price', xaxis: { title: 'Store' }, yaxis: { title: 'Price' } };
    plot(scatterData, scatterLayout);

    // Box plot of price
    const priceData = [{ y: priceUSD, type: 'box' }];
    const priceLayout = { title: 'Price Distribution' };
    plot(priceData, priceLayout);

    // Bar chart of a categorical feature
    const brandCounts = _countUniqueLabels(brand);
    const locCounts = _countUniqueLabels(storeLocation);
    const brandLabels = Object.keys(brandCounts);
    const locLabels = Object.keys(locCounts);
    const brandData = brandLabels.map(label => brandCounts[label]);
    const locData = locLabels.map(label => locCounts[label]);
    const brandBar = [{ x: brandLabels, y: brandData, type: 'bar' }];
    const locBar = [{ x: locLabels, y: locData, type: 'bar' }];
    const brandLayout = { title: 'Brand Distribution' };
    const locLayout = { title: 'Location Distribution' };
    plot(locBar, brandLayout);
    plot(brandBar, locLayout);

    // Line chart of price over time (Best before date)
    const priceOverTime = bestBeforeDate.map((date, index) => ({ x: date, y: priceUSD[index] }));
    priceOverTime.sort((a, b) => a.x - b.x); // Sort by date in ascending order
    const lineData = [{ x: priceOverTime.map(entry => entry.x), y: priceOverTime.map(entry => entry.y), type: 'scatter' }];
    const lineLayout = { title: 'Price Over Time', xaxis: { type: 'date' }, yaxis: { title: 'Price' } };

    plot(lineData, lineLayout);
}




/**
 * Normalize to their relative wieghts.
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
    let min = null;
    let max = null;

    // We will need to normalize input data with the same constants.
    if (!metaData[`${featureName}_norm`]) {
        min = tf.min(feature);
        max = tf.max(feature);
        metaData[`${featureName}_norm`] = { min: min.dataSync()[0], max: max.dataSync()[0] };
    } else {
        min = metaData[`${featureName}_norm`].min;
        max = metaData[`${featureName}_norm`].max;
    }

    const normalizedFeature = tf.div(tf.sub(feature, min), tf.sub(max, min));

    return normalizedFeature;
}

/**
 * Scale to a range of large magnitude.
 * 
 * @see DATASETS_METADATA
 * @returns {Array[*]} The scaled tensor.
 */
function scaleFeature(feature, featureName, metaData = DATASETS_METADATA) {
    let mean = null;
    let std = null;

    // We will need to normalize input data with the same constants.
    if (!metaData[`${featureName}_norm`]) {
        mean = tf.mean(feature);
        std = tf.moments(feature).variance.sqrt().arraySync();
        metaData[`${featureName}_norm`] = { mean: mean, std: std };
    } else {
        mean = metaData[`${featureName}_norm`].mean;
        std = metaData[`${featureName}_norm`].std;
    }

    const scaledFeature = tf.div(tf.sub(feature, mean), std);

    return scaledFeature;
}

/**
 * Cleans, nromalizes and drops irrelavant data. Then splits the data into train, validate, test sets.
 * 
 * @param {*} data 
 * @param {*} trainRatio 
 * @param {*} testRatio 
 * @param {*} valRatio 
 * @returns {Object} of: {
 *      trainData: {Tensor},
 *      testData: {Tensor},
 *      validationData: {Tensor}
 *   }
 */
function cleanTrainSpitData(data, trainRatio = 0.7, testRatio = 0.1, valRatio = 0.2) {

    // Remove irrelevant features (date in this case) and NaNs
    const cleanedAndNormalizedData = { weight: [], brandOHE: [], storeOHE: [], priceUSD: [] };

    for (let i = 0; i < data.weight.length; i++) {
        // Handle missing values if needed
        if (!isNaN(data.weight[i]) && !isNaN(data.priceUSD[i]) && (data.brand[i])) {
            cleanedAndNormalizedData.weight.push(data.weight[i]);
            cleanedAndNormalizedData.brandOHE.push(data.brand[i]);
            cleanedAndNormalizedData.priceUSD.push(data.priceUSD[i]);
        }
    }

    // Normalize the Data
    cleanedAndNormalizedData.weight = normalizeFeature(cleanedAndNormalizedData.weight, 'weight');
    cleanedAndNormalizedData.brandOHE = oneHotEncode(cleanedAndNormalizedData.brandOHE, BRANDS, 'brand');
    cleanedAndNormalizedData.priceUSD = tf.tensor1d(cleanedAndNormalizedData.priceUSD);

    const { weight, brandOHE, storeOHE, priceUSD } = cleanedAndNormalizedData;
    const totalSize = weight.shape[0];
    const trainIndex = Math.floor(trainRatio * totalSize);
    const valSize = Math.floor(valRatio * totalSize);
    const testIndex = trainIndex + valSize;

    const trainData = {
        weight: weight.slice([0], [trainIndex]),
        brandOHE: brandOHE.slice([0], [trainIndex]),
        priceUSD: priceUSD.slice([0], [trainIndex])
    };
    const validationData = {
        weight: weight.slice([trainIndex], [valSize]),
        brandOHE: brandOHE.slice([trainIndex], [valSize]),
        priceUSD: priceUSD.slice([trainIndex], [valSize])
    };
    const testData = {
        weight: weight.slice([testIndex]),
        brandOHE: brandOHE.slice([testIndex]),
        priceUSD: priceUSD.slice([testIndex])
    };

    return {
        trainData: trainData,
        testData: testData,
        validationData: validationData
    };
}

/**
 * 
 * @param {*} trainData 
 * @param {*} validationData 
 * @param {*} testData 
 * @param {*} numEpochs 
 */
async function buildLinearRegressionModel(trainData, validationData, testData, epochs) {
    const { weight, brandOHE, storeOHE, priceUSD } = trainData;
    const trainX = tf.tensor2d(
        tf.concat([
            tf.tensor2d(weight.arraySync(), [weight.arraySync().length, 1]),
            tf.tensor2d(brandOHE.arraySync())], 1)
            .arraySync());
    const trainY = tf.tensor1d(priceUSD.arraySync());

    console.log('trainX shape:', trainX.shape);
    console.log('trainY shape:', trainY.shape);

    const model = tf.sequential();
    model.add(tf.layers.dense({
        units: 32,
        activation: 'relu',
        inputShape: [trainX.shape[1]]
    }));
    model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 8, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'linear' }));
    model.compile({
        optimizer: 'adam',
        loss: 'meanSquaredError',
        metrics: ['mse', 'mae']
    });

    const history = await model.fit(trainX, trainY, { validationData: validationData, epochs: epochs });

    console.log("Model trained and fitted!")

    const { weight: testWeight, brandOHE: testBrandOHE, storeOHE: testStoreOHE, priceUSD: testPriceUSD } = testData;

    const testX = tf.tensor2d(
        tf.concat([
            tf.tensor2d(testWeight.arraySync(), [testWeight.arraySync().length, 1]),
            tf.tensor2d(testBrandOHE.arraySync())], 1)
            .arraySync());
    const testY = tf.tensor1d(testPriceUSD.arraySync());

    console.log('testX shape:', testX.shape);
    console.log('testY shape:', testY.shape);

    const testPredictions = await model.predict(testX);

    return {
        model: model,
        predictions: testPredictions,
        trueValues: testY,
        history: history.history
    };
}

/**
 * 
 * @param {*} model 
 * @param {*} testData 
 */
async function modelMetrics(modelMetaData) {
    // Calculate the mean absolute error (MAE)
    const maeTensor = tf.metrics.meanAbsoluteError(modelMetaData.trueValues, modelMetaData.predictions);
    const maeValue = await maeTensor.data();
    console.log('Mean Absolute Error (MAE):', maeValue[0]);

    // Calculate the mean squared error (MSE)
    const mseTensor = tf.metrics.meanSquaredError(modelMetaData.trueValues, modelMetaData.predictions);
    const mseValue = await mseTensor.data();
    console.log('Mean Squared Error (MSE):', mseValue[0]);
}


/**
 * Main entry.
 * 
 * Doesn't return promises so ts.tidy can clean up memory.
 */
function main() {
    (async () => {
        console.log('Generating Synth Data');
        const catFoodDataset = await generateData(MAX_DS_X);
        await dataEDA(catFoodDataset); // For EDA only.

        console.log('Clean and Split Data');
        const datasets = await cleanTrainSpitData(catFoodDataset);

        console.log('Build Model');
        const modelMetaData = await buildLinearRegressionModel(datasets.trainData, datasets.validationData, datasets.trainData, EPOCHS);

        console.log('Get Model Metrics');
        await modelMetrics(modelMetaData, datasets.trainData);

        console.log('Testing the model on inputs');
        const brand = oneHotEncode([BRANDS[1]], BRANDS, 'brand');
        const wieghtInGrams = 5000;
        const wieght = normalizeFeature([wieghtInGrams], 'weight');

        const x = tf.tensor2d(
            tf.concat([
                tf.tensor2d(wieght.arraySync(), [wieght.arraySync().length, 1]),
                tf.tensor2d(brand.arraySync())], 1)
                .arraySync());

        const prediciton = await modelMetaData.model.predict(x);

        console.log(`Predicted: '$${prediciton.dataSync()}' for a brand: '${BRANDS[1]}' and weight: '${wieghtInGrams}g'`);
        console.log(`Saving the model to the firebase function dir: ${FUNCTION_MODEL_PATH}`);
        await modelMetaData.model.save(tf.io.fileSystem(FUNCTION_MODEL_PATH));

        let modelMetaJSON = JSON.stringify(DATASETS_METADATA);
        fs.writeFileSync(`${FUNCTION_MODEL_PATH}/meta.json`, modelMetaJSON);
    })();
}

// Protect everything with a tiny memory manager.
// Avoid any Promise() return!
tidy(() => main());