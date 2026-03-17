const http = require('http');

const BASE_URL = "http://localhost:3000";

function test_post() {
    const data = {
        text: "Hello, welcome to TalkJourney! This is a test of the text-to-speech API.",
        languageCode: "en-US",
        voiceName: "en-US-Standard-I"
    };

    const postData = JSON.stringify(data);

    const options = {
        hostname: 'localhost',
        port: 3000,
        path: '/api/get-audio',
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Content-Length': Buffer.byteLength(postData)
        }
    };

    const req = http.request(options, (res) => {
        let responseData = '';

        res.on('data', (chunk) => {
            responseData += chunk;
        });

        res.on('end', () => {
            console.log("POST Response:", JSON.parse(responseData));
        });
    });

    req.on('error', (e) => {
        console.error(`Problem with request: ${e.message}`);
    });

    req.write(postData);
    req.end();
}

test_post();