# Firebase CloudFunctions and TensorflowJS - Serve your Model on Budget

In this article we will utilize Firebase Cloud Functions project to host a TensorFlow.js model and open it up to predictions through API calls. 
We'll save call telemetry to the Firebase Realtime Database, to give us history we can analyze later on.

# Start with Firebase

Register your free firebase account here: https://console.firebase.google.com/. 
You will have to create a project, keep it's name in mind.

Then on your machine, install firebase globally:

`npm install -g firebase-tools`

Login to the account your have created:

`firebase login`

And initialize your project by selecting the existing project you created, cloud firebase capabilities, cloud realtime database serivce and the emulator. Use the follow command:

`firebase init`

You'll follow the prompts and enable the services mentioned above. At the end of the whole process, you will have a *firebase.json* file that is similar to this:

```json
{
  "database": {
    "rules": "database.rules.json"
  },
  "functions": [
    {
      "source": "functions",
      "codebase": "default",
      "ignore": [
        "node_modules",
        ".git",
        "firebase-debug.log",
        "firebase-debug.*.log"
      ]
    }
  ],
  "emulators": {
    "functions": {
      "port": 5001
    },
    "database": {
      "port": 9000
    },
    "ui": {
      "enabled": true
    },
    "singleProjectMode": true
  }
}
```

# Create the NodeJs cloud Function

The firebase cli tool should have created a **functions** folder, navigate to the folder.

Edit *index.js* with these details to create the entry script:

```javascript
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

```

install all the imported libraries:

`npm install express firebase-admin`

# Emulator to Test the Function

In our project setup, we asked for an emulator to validate our code before going to the cloud. Let's run it:

`firebase emulators:start`

it will print the Emulator's UI url which you can use to browse to:

![alt](./article/enulatorCli.PNG)

Go to the function, and access the given url. You should see the 'hello world' ouput, and the logs should start showing on the UI:

![alt](./article/emulatorFunction.PNG)

Now let's access the DB's emulator UI and check what telemetery we go:

![alt](./article/emulatorDB.PNG)

With the emulator validating our setup, let's deploy to the actual Firebase service

# Hello Firebase

To deploy our code, we type the following in the Cli:

`firebase deploy`

Don't worry if you are asked to switch the payment plan to blaze (mostly because of the cloud functions) and enter your credit card details.
If the deploy is successful, you should see this in your cli:

![alt](./article/deployCli.PNG)


The cloud function should be visible from firebase:

![alt](./article/fbFunction.PNG)

If you curl or use Postman to hit that url show on firebase, you will get hello world:

![alt](./article/curlHelloWorld.PNG)

# Show us the Models!

Time to step it up with some datascience. 

It's recommended to create a folder where we will process the data and test the model, before embedding it into the cloud function. CD to the folder, and run `npm init` to initialize a simple package setup. 

From here install all required TensorFlowJS dependencies:

`npm install @tensorflow/tfjs @tensorflow/tfjs-node`

For this experiment, we don't have any dataset - so let's synthesize one. We want data that describes catfood, and we will use this to find the best price of a purchase.
