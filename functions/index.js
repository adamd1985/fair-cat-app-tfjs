/**
 * Cloud function
 * @adamd1985
 * @see https://firebase.google.com/docs/functions
 */

const { onRequest } = require("firebase-functions/v2/https");
const logger = require("firebase-functions/logger");
const admin = require('firebase-admin');

admin.initializeApp();
const database = admin.database();

let GLOBAL_COUNT = 0;

exports.helloWorld = onRequest(async (request, response) => {
    logger.info("Hello logs!", { structuredData: true });

    // Save telemetry to Firebase Realtime Database
    await database.ref('telemetry').push({
        msg: `Hello #${GLOBAL_COUNT++}`,
        timestamp: Date.now(),
    });

    response.send("Hello from Firebase!");
});
