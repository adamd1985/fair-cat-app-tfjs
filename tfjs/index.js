/**
 * Tensorflow JS Analysis and Model Building.
 * @adamd1985
 */

import * as tf from '@tensorflow/tfjs-node'
import { plot } from 'nodeplotlib';
import Plot from 'nodeplotlib';
const { tidy, tensor2d } = tf;

// Constants
const BRANDS = ['Unbranded', 'Whiskers and Paws', 'Royal Feline', 'Meowarf'];
const STORES = ['Fresh Pet', 'Expensive Cats', 'Overpriced Pets', 'Jungle of Money', 'Mom & Pop Petshop'];
const MAX_DS_X = 1000;
const EPOCHS = 30;

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
function oneHotEncode(labels) {
    const uniqueLabels = Array.from(new Set(labels));
    const numCategories = uniqueLabels.length;

    const encodedLabels = labels.map(label => uniqueLabels.indexOf(label));
    const encodedTensor = tf.oneHot(tf.tensor1d(encodedLabels, 'int32'), numCategories);

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

// Const metadata for new input data.
const DATASETS_METADATA = {

};

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

    /**
     * local function to noramlize a range, will save the mins and maxs to a global cache to be used in a prediction.
     * 
     * @see MINIMUMS
     * @returns {Array[*]} The normalized range.
     */
    function _normalizeFeature(feature, featureName, metaData = DATASETS_METADATA) {
        const min = tf.min(feature);
        const max = tf.max(feature);
        const normalizedFeature = tf.div(tf.sub(feature, min), tf.sub(max, min));

        // We will need to normalize input data with the same constants.
        metaData[featureName] = { min: min, max: max };

        return normalizedFeature;
    }

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
    cleanedAndNormalizedData.weight = _normalizeFeature(cleanedAndNormalizedData.weight, 'weight');
    cleanedAndNormalizedData.brandOHE = oneHotEncode(cleanedAndNormalizedData.brandOHE);
    cleanedAndNormalizedData.priceUSD = _normalizeFeature(cleanedAndNormalizedData.priceUSD, 'priceUSD');

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
        units: trainX.shape[0],
        activation: 'sigmoid',
        inputShape: [trainX.shape[1]]
    }));
    model.add(tf.layers.dense({ units: trainX.shape[0] / 2, activation: 'sigmoid' }));
    model.add(tf.layers.dense({ units: 1, activation: 'linear' }));
    model.compile({
        optimizer: 'adam',
        loss: 'meanSquaredError',
        metrics: ['accuracy']
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
    const accuracy = tf.metrics.binaryAccuracy(modelMetaData.trueValues, modelMetaData.predictions);
    const error = tf.metrics.meanAbsoluteError(modelMetaData.trueValues, modelMetaData.predictions);

    console.log(`Accuracy: ${accuracy.arraySync()[accuracy.arraySync().length - 1] * 100}%`);
    console.log(`Error: ${error.arraySync()[error.arraySync().length - 1] * 100}%`);

    console.log(`Loss: ${[modelMetaData.history.loss.length - 1]}%`);
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
    })();
}

// Protect everything with a tiny memory manager.
// Avoid any Promise() return!
tidy(() => main());