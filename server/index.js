require('dotenv').config();
const express = require('express');
const admin = require('firebase-admin');
const textToSpeech = require('@google-cloud/text-to-speech');
const crypto = require('crypto');
const PORT = 3000;

function log(message, level = 'info') {
    const time = new Date();
    const timestamp = time.toLocaleDateString('en-IL') + ' ' + time.toLocaleTimeString('en-IL');        
    console[level](`[${timestamp}] ${message}`);
}


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
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

async function verifyToken(req, res, next) {
    log('\x1b[33m[NEW]\x1b[0m Received request, verifying token...');
    const authHeader = req.headers.authorization;
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
        log('\x1b[31m[ERR]\x1b[0m Blocked request: Missing or invalid Authorization header', 'warn');
        return res.status(401).json({ error: 'Unauthorized: No token provided' });
    }

    const idToken = authHeader.split('Bearer ')[1];

    try {
        // Verify the token using Firebase Admin
        const decodedToken = await admin.auth().verifyIdToken(idToken);
        log('Token verified successfully, user ID: ' + decodedToken.uid);
        req.user = decodedToken; // Save user data for later if needed (e.g., req.user.uid)
        next(); // Token is valid, proceed to the API logic
    } catch (error) {
        log('\x1b[31m[ERR]\x1b[0m Blocked request: Invalid token', 'error');
        return res.status(401).json({ error: 'Unauthorized: Invalid token' });
    }
}

app.post('/api/get-audio', verifyToken, async (req, res) => {
    const { text, languageCode = 'en-US', voiceName=null } = req.body;
    log('Received request for audio retrieval from user: ' + req.user.uid);
    if (!text) {
        return res.status(400).json({ error: 'Text parameter is required' });
    }
    var voice;
    if (!voiceName) {
        voice = languageCode+'-Standard-A';
    }
    else {
        voice = voiceName;
    }

    const hash = crypto.createHash('sha256').update(text + languageCode + voice).digest('hex');
    const fileName = `audio-cache/${hash}.mp3`;
    const file = bucket.file(fileName);

    try {
        const [exists] = await file.exists();

        if (exists) {
            log('\x1b[32m[END]\x1b[0m File found in cache.');
            const url = await getDownloadUrl(file);
            return res.json({ url: url, isCached: true });
        }

        log('Generating new audio via TTS.');
        const request = {
            input: { text: text },
            voice: { languageCode: languageCode, name: voice },
            audioConfig: { audioEncoding: 'MP3' },
        };

        const [response] = await ttsClient.synthesizeSpeech(request);
        await file.save(response.audioContent, {
            metadata: { contentType: 'audio/mpeg' },
        });

        log('\x1b[32m[END]\x1b[0m Audio saved to Firebase Storage.');
        const url = await getDownloadUrl(file);
        return res.json({ url: url, isCached: false });

    } catch (error) {
        log('\x1b[31m[ERR]\x1b[0m Error:', 'error');
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
    log(`Server running on port ${PORT}`);
});


const speech = require('@google-cloud/speech');

const sttClient = new speech.SpeechClient({
    credentials: {
        client_email: process.env.FIREBASE_CLIENT_EMAIL,
        private_key: process.env.FIREBASE_PRIVATE_KEY,
    },
    projectId: process.env.FIREBASE_PROJECT_ID,
});

app.post('/api/transcribe', verifyToken, async (req, res) => {
    try {
        const { audioBase64, languageCode = 'en-US' } = req.body;

        if (!audioBase64) {
            return res.status(400).json({ error: 'audioBase64 required' });
        }

        const [response] = await sttClient.recognize({
            audio: { content: audioBase64 },
            config: {
                encoding: 'LINEAR16',
                sampleRateHertz: 16000,
                languageCode,
                enableAutomaticPunctuation: true,
            },
        });

        const transcript = (response.results || [])
            .map(r => r.alternatives?.[0]?.transcript || '')
            .join(' ')
            .trim();

        return res.json({ transcript });
    } catch (error) {
        log('[ERR] STT failed', 'error');
        return res.status(500).json({ error: error.message || 'Internal Server Error' });
    }
});