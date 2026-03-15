using System;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Networking;

/*
    gets the audio from the back end
    if appendIdentifierAsPath = true
        url = {baseUrl}/{id}
    else
        url = {baseUrl}?{id}={value}
*/

namespace TalkJourney.BubbleSystem.Audio
{
    [DisallowMultipleComponent]
    public class AudioBackendClient : MonoBehaviour, IAudioBackendClient
    {
        [Header("Backend Endpoint")]
        [Tooltip("Base URL for the custom backend. Example: https://api.example.com/audio")]
        public string baseUrl = "http://localhost:5000/audio";

        [Tooltip("If true, the audio identifier is appended as a path segment: {baseUrl}/{id}")]
        public bool appendIdentifierAsPath = true;

        [Tooltip("If appendIdentifierAsPath is false, identifier is sent as query param name below.")]
        public string queryParameterName = "id";

        [Header("Request Settings")]
        [Tooltip("Expected response audio format.")]
        public AudioType responseAudioType = AudioType.WAV;

        [Min(1)]
        [Tooltip("Network timeout for each audio request.")]
        public int timeoutSeconds = 15;

        public async Task<AudioRequestResult> RequestAudioAsync(string audioIdentifier, CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrWhiteSpace(audioIdentifier))
            {
                return AudioRequestResult.Failure("Audio identifier is missing.");
            }

            if (string.IsNullOrWhiteSpace(baseUrl))
            {
                return AudioRequestResult.Failure("Audio backend base URL is not configured.");
            }

            var requestUrl = BuildRequestUrl(audioIdentifier.Trim());

            using (var request = UnityWebRequestMultimedia.GetAudioClip(requestUrl, responseAudioType))
            {
                request.timeout = timeoutSeconds;
                var operation = request.SendWebRequest();

                while (!operation.isDone)
                {
                    if (cancellationToken.IsCancellationRequested)
                    {
                        request.Abort();
                        return AudioRequestResult.Failure("Audio request was cancelled.");
                    }

                    await Task.Yield();
                }

                if (request.result != UnityWebRequest.Result.Success)
                {
                    return AudioRequestResult.Failure($"Audio request failed: {request.error}");
                }

                var clip = DownloadHandlerAudioClip.GetContent(request);
                if (clip == null)
                {
                    return AudioRequestResult.Failure("Audio response was empty.");
                }

                return AudioRequestResult.Success(clip);
            }
        }

        private string BuildRequestUrl(string audioIdentifier)
        {
            var encoded = UnityWebRequest.EscapeURL(audioIdentifier);

            if (appendIdentifierAsPath)
            {
                return $"{baseUrl.TrimEnd('/')}/{encoded}";
            }

            var separator = baseUrl.Contains("?") ? "&" : "?";
            return $"{baseUrl}{separator}{queryParameterName}={encoded}";
        }
    }
}
