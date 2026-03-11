require('dotenv').config();
const express = require('express');
const admin = require('firebase-admin');
const textToSpeech = require('@google-cloud/text-to-speech');
const crypto = require('crypto');
const PORT = 3000;

const privateKey = process.env.FIREBASE_PRIVATE_KEY
const credentials = {
    projectId: process.env.FIREBASE_PROJECT_ID,
    clientEmail: process.env.FIREBASE_CLIENT_EMAIL,
    privateKey: privateKey
};

admin.initializeApp({
    credential: admin.credential.cert(credentials),
    storageBucket: process.env.FIREBASE_STORAGE_BUCKET
});
const bucket = admin.storage().bucket();

const ttsClient = new textToSpeech.TextToSpeechClient({
    credentials: {
        client_email: credentials.clientEmail,
        private_key: credentials.privateKey
    },
    projectId: credentials.projectId
});
const app = express();
app.use(express.json());

app.post('/api/get-audio', async (req, res) => {
    const { text, languageCode = 'en-US', voiceName='en-US-Standard-A' } = req.body;

    if (!text) {
        return res.status(400).json({ error: 'Text parameter is required' });
    }

    const hash = crypto.createHash('sha256').update(text + languageCode + voiceName).digest('hex');
    const fileName = `audio-cache/${hash}.mp3`;
    const file = bucket.file(fileName);

    try {
        const [exists] = await file.exists();

        if (exists) {
            console.log('File found in cache.');
            const url = await getDownloadUrl(file);
            return res.json({ url: url, isCached: true });
        }

        console.log('Generating new audio via TTS.');
        const request = {
            input: { text: text },
            voice: { languageCode: languageCode, name: voiceName },
            audioConfig: { audioEncoding: 'MP3' },
        };

        const [response] = await ttsClient.synthesizeSpeech(request);
        await file.save(response.audioContent, {
            metadata: { contentType: 'audio/mpeg' },
        });

        console.log('Audio saved to Firebase Storage.');
        const url = await getDownloadUrl(file);
        return res.json({ url: url, isCached: false });

    } catch (error) {
        console.error('Error:', error);
        return res.status(500).json({ error: 'Internal Server Error' });
    }
});
async function getDownloadUrl(file) {
    const [url] = await file.getSignedUrl({
        action: 'read',
        expires: Date.now() + 2 * 60 * 60 * 1000 
    });
    return url;
}

app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});