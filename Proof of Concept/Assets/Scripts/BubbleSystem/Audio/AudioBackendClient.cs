using System;
using System.Threading;
using System.Threading.Tasks;
using Firebase;
using Firebase.Auth;
using TalkJourney.GameServices.Auth;
using TalkJourney.BubbleSystem.Localization;
using UnityEngine;
using UnityEngine.Networking;

namespace TalkJourney.BubbleSystem.Audio
{
    [DisallowMultipleComponent]
    public class AudioBackendClient : MonoBehaviour, IAudioBackendClient
    {
        [Serializable]
        private class TtsAudioRequest
        {
            public string text;
            public string languageCode;
            public string voiceName;
        }

        [Serializable]
        private class TtsAudioResponse
        {
            public string url;
            public bool isCached;
        }

        [Header("Backend Endpoint")]
        [Tooltip("TTS POST endpoint. Example: http://localhost:3000/api/get-audio")]
        public string ttsPostUrl = "http://localhost:3000/api/get-audio";

        [Header("Request Settings")]
        [Tooltip("Expected response audio format.")]
        public AudioType responseAudioType = AudioType.MPEG;

        [Tooltip("If enabled, infer audio format from the file URL extension (mp3, wav, ogg) before decoding.")]
        public bool inferAudioTypeFromUrl = true;

        [Tooltip("Language code sent with TTS request body.")]
        public string languageCode = "en-US";

        [Tooltip("Voice name sent with TTS request body.")]
        public string voiceName = "en-US-Standard-A";

        [Tooltip("Optional bearer token used for Authorization header. Leave empty if backend does not require auth.")]
        public string bearerToken;

        [Tooltip("Optional component implementing IAuthTokenProvider. If assigned, it is used for auth tokens before local fallback.")]
        public MonoBehaviour authTokenProviderBehaviour;

        [Tooltip("When enabled, uses authTokenProviderBehaviour first (if available).")]
        public bool preferExternalAuthTokenProvider = true;

        [Tooltip("When enabled, fetches Firebase ID token for Authorization header.")]
        public bool useFirebaseAuthToken = true;

        [Tooltip("If true, refreshes Firebase token before each request.")]
        public bool forceRefreshFirebaseToken = true;

        [Tooltip("If enabled, performs anonymous sign-in when no Firebase user is available.")]
        public bool signInAnonymouslyIfNeeded = true;

        [Min(1)]
        [Tooltip("Network timeout for each audio request.")]
        public int timeoutSeconds = 15;

        [Tooltip("Optional LocalizationResolver. If empty, one is found automatically.")]
        public LocalizationResolver localizationResolver;

        private IAuthTokenProvider _authTokenProvider;

        private void OnEnable()
        {
            ResolveDependencies();
            LocalizationResolver.OnDisplayLanguageChanged += OnDisplayLanguageChanged;

            if (localizationResolver != null)
            {
                ApplyVoiceSettingsForDisplayLanguage(localizationResolver.selectedLanguage);
            }
        }

        private void OnDisable()
        {
            LocalizationResolver.OnDisplayLanguageChanged -= OnDisplayLanguageChanged;
        }

        public async Task<AudioRequestResult> RequestAudioFromTextAsync(string text, CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrWhiteSpace(text))
            {
                return AudioRequestResult.Failure("TTS text is missing.");
            }

            if (string.IsNullOrWhiteSpace(ttsPostUrl))
            {
                return AudioRequestResult.Failure("TTS POST URL is not configured.");
            }

            var requestPayload = new TtsAudioRequest
            {
                text = text.Trim(),
                languageCode = languageCode,
                voiceName = voiceName
            };

            var resolvedBearerToken = await ResolveAuthorizationTokenAsync(cancellationToken);

            var payloadJson = JsonUtility.ToJson(requestPayload);

            using (var postRequest = new UnityWebRequest(ttsPostUrl, UnityWebRequest.kHttpVerbPOST))
            {
                var bodyRaw = System.Text.Encoding.UTF8.GetBytes(payloadJson);
                postRequest.uploadHandler = new UploadHandlerRaw(bodyRaw);
                postRequest.downloadHandler = new DownloadHandlerBuffer();
                postRequest.timeout = timeoutSeconds;
                postRequest.SetRequestHeader("Content-Type", "application/json");

                if (!string.IsNullOrWhiteSpace(resolvedBearerToken))
                {
                    postRequest.SetRequestHeader("Authorization", "Bearer " + resolvedBearerToken.Trim());
                }

                var postOperation = postRequest.SendWebRequest();

                while (!postOperation.isDone)
                {
                    if (cancellationToken.IsCancellationRequested)
                    {
                        postRequest.Abort();
                        return AudioRequestResult.Failure("TTS request was cancelled.");
                    }

                    await Task.Yield();
                }

                if (postRequest.result != UnityWebRequest.Result.Success)
                {
                    if (postRequest.responseCode == 401)
                    {
                        return AudioRequestResult.Failure("TTS request failed: 401 Unauthorized. Provide a valid bearerToken if your backend requires auth.");
                    }

                    return AudioRequestResult.Failure($"TTS request failed: {postRequest.error}");
                }

                var responseJson = postRequest.downloadHandler.text;
                if (string.IsNullOrWhiteSpace(responseJson))
                {
                    return AudioRequestResult.Failure("TTS response was empty.");
                }

                var response = JsonUtility.FromJson<TtsAudioResponse>(responseJson);
                if (response == null || string.IsNullOrWhiteSpace(response.url))
                {
                    return AudioRequestResult.Failure("TTS response did not contain an audio URL.");
                }

                return await DownloadClipFromUrlAsync(response.url, cancellationToken);
            }
        }

        private async Task<AudioRequestResult> DownloadClipFromUrlAsync(string fileUrl, CancellationToken cancellationToken)
        {
            var resolvedAudioType = ResolveAudioType(fileUrl);

            using (var audioRequest = UnityWebRequestMultimedia.GetAudioClip(fileUrl, resolvedAudioType))
            {
                audioRequest.timeout = timeoutSeconds;
                audioRequest.SetRequestHeader("Accept", "audio/*");
                var audioOperation = audioRequest.SendWebRequest();

                while (!audioOperation.isDone)
                {
                    if (cancellationToken.IsCancellationRequested)
                    {
                        audioRequest.Abort();
                        return AudioRequestResult.Failure("Audio download was cancelled.");
                    }

                    await Task.Yield();
                }

                if (audioRequest.result != UnityWebRequest.Result.Success)
                {
                    return AudioRequestResult.Failure($"Audio download failed: {audioRequest.error}");
                }

                var contentType = audioRequest.GetResponseHeader("Content-Type");
                if (!string.IsNullOrWhiteSpace(contentType)
                    && !contentType.StartsWith("audio/", StringComparison.OrdinalIgnoreCase))
                {
                    return AudioRequestResult.Failure($"Audio download returned non-audio content-type '{contentType}' from URL '{fileUrl}'.");
                }

                AudioClip clip;
                try
                {
                    clip = DownloadHandlerAudioClip.GetContent(audioRequest);
                }
                catch (Exception exception)
                {
                    return AudioRequestResult.Failure($"Audio decode failed for URL '{fileUrl}' using type '{resolvedAudioType}': {exception.Message}");
                }

                if (clip == null)
                {
                    return AudioRequestResult.Failure($"Downloaded audio clip was empty for URL '{fileUrl}' using type '{resolvedAudioType}'.");
                }

                return AudioRequestResult.Success(clip);
            }
        }

        private AudioType ResolveAudioType(string fileUrl)
        {
            if (!inferAudioTypeFromUrl || string.IsNullOrWhiteSpace(fileUrl))
            {
                return responseAudioType;
            }

            var trimmedUrl = fileUrl.Trim();
            var queryIndex = trimmedUrl.IndexOf('?');
            if (queryIndex >= 0)
            {
                trimmedUrl = trimmedUrl.Substring(0, queryIndex);
            }

            var lowerUrl = trimmedUrl.ToLowerInvariant();
            if (lowerUrl.EndsWith(".mp3") || lowerUrl.EndsWith(".mpeg"))
            {
                return AudioType.MPEG;
            }

            if (lowerUrl.EndsWith(".wav"))
            {
                return AudioType.WAV;
            }

            if (lowerUrl.EndsWith(".ogg"))
            {
                return AudioType.OGGVORBIS;
            }

            return responseAudioType;
        }

        private async Task<string> ResolveAuthorizationTokenAsync()
        {
            if (preferExternalAuthTokenProvider && _authTokenProvider != null)
            {
                var externalToken = await _authTokenProvider.GetAuthorizationTokenAsync();
                if (!string.IsNullOrWhiteSpace(externalToken))
                {
                    return externalToken;
                }
            }

            if (useFirebaseAuthToken)
            {
                var firebaseToken = await TryGetFirebaseTokenAsync();
                if (!string.IsNullOrWhiteSpace(firebaseToken))
                {
                    return firebaseToken;
                }
            }

            return bearerToken;
        }

        private async Task<string> ResolveAuthorizationTokenAsync(CancellationToken cancellationToken)
        {
            if (preferExternalAuthTokenProvider && _authTokenProvider != null)
            {
                var externalToken = await _authTokenProvider.GetAuthorizationTokenAsync(cancellationToken);
                if (!string.IsNullOrWhiteSpace(externalToken))
                {
                    return externalToken;
                }
            }

            if (cancellationToken.IsCancellationRequested)
            {
                return null;
            }

            return await ResolveAuthorizationTokenAsync();
        }

        private void ResolveDependencies()
        {
            if (localizationResolver == null)
            {
                localizationResolver = FindFirstObjectByType<LocalizationResolver>(FindObjectsInactive.Include);
            }

            _authTokenProvider = authTokenProviderBehaviour as IAuthTokenProvider;
            if (_authTokenProvider == null && authTokenProviderBehaviour == null)
            {
                var sceneBehaviours = FindObjectsByType<MonoBehaviour>(FindObjectsInactive.Include, FindObjectsSortMode.None);
                for (int i = 0; i < sceneBehaviours.Length; i++)
                {
                    var behaviour = sceneBehaviours[i];
                    if (behaviour is IAuthTokenProvider sceneProvider)
                    {
                        _authTokenProvider = sceneProvider;
                        authTokenProviderBehaviour = behaviour;
                        break;
                    }
                }
            }
        }

        private void OnDisplayLanguageChanged(DisplayLanguage language)
        {
            ApplyVoiceSettingsForDisplayLanguage(language);
        }

        private void ApplyVoiceSettingsForDisplayLanguage(DisplayLanguage language)
        {
            switch (language)
            {
                case DisplayLanguage.English:
                    languageCode = "en-US";
                    voiceName = "en-US-Standard-A";
                    break;
                case DisplayLanguage.Hebrew:
                    languageCode = "he-IL";
                    voiceName = "he-IL-Standard-A";
                    break;
                case DisplayLanguage.Russian:
                    languageCode = "ru-RU";
                    voiceName = "ru-RU-Standard-A";
                    break;
                default:
                    languageCode = "en-US";
                    voiceName = "en-US-Standard-A";
                    break;
            }
        }

        private async Task<string> TryGetFirebaseTokenAsync()
        {
            try
            {
                var dependencyStatus = await FirebaseApp.CheckAndFixDependenciesAsync();
                if (dependencyStatus != DependencyStatus.Available)
                {
                    Debug.LogWarning($"Firebase dependencies unavailable: {dependencyStatus}", this);
                    return null;
                }

                var auth = FirebaseAuth.DefaultInstance;
                var user = auth.CurrentUser;

                if (user == null && signInAnonymouslyIfNeeded)
                {
                    var signInResult = await auth.SignInAnonymouslyAsync();
                    user = signInResult?.User;
                }

                if (user == null)
                {
                    return null;
                }

                return await user.TokenAsync(forceRefreshFirebaseToken);
            }
            catch (Exception exception)
            {
                Debug.LogWarning($"Firebase token fetch failed: {exception.Message}", this);
                return null;
            }
        }
    }
}
