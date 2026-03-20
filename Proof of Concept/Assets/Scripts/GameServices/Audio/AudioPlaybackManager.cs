using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using TalkJourney.BubbleSystem.Events;

namespace TalkJourney.BubbleSystem.Audio
{
    public enum OverlapPolicy
    {
        StopCurrent,
        SkipIfBusy,
        Queue
    }

    [DisallowMultipleComponent]
    public class AudioPlaybackManager : MonoBehaviour, IAudioPlaybackManager
    {
        [Header("Dependencies")]
        [Tooltip("Audio source used for all bubble and sentence playback.")]
        public AudioSource playbackSource;

        [Tooltip("Component that implements IAudioBackendClient.")]
        public MonoBehaviour backendClientBehaviour;

        [Header("Playback")]
        public OverlapPolicy overlapPolicy = OverlapPolicy.StopCurrent;

        [Tooltip("If true, fetched clips are stored by identifier to avoid repeated backend requests.")]
        public bool enableCaching = true;

        private readonly Dictionary<string, AudioClip> _cache = new Dictionary<string, AudioClip>();
        private readonly object _queueLock = new object();
        private Task _queueTail = Task.CompletedTask;
        private IAudioBackendClient _backendClient;

        private void Awake()
        {
            RefreshDependencies();
        }

        public void RefreshDependencies()
        {
            _backendClient = backendClientBehaviour as IAudioBackendClient;

            if (_backendClient == null)
            {
                Debug.LogError("AudioPlaybackManager requires backendClientBehaviour implementing IAudioBackendClient.", this);
            }
        }

        public Task<bool> PlayByTextAsync(string text, CancellationToken cancellationToken = default)
        {
            if (_backendClient == null)
            {
                return Task.FromResult(false);
            }

            if (overlapPolicy == OverlapPolicy.Queue)
            {
                return EnqueuePlayback(() => PlayByTextInternalAsync(text, cancellationToken));
            }

            return PlayByTextInternalAsync(text, cancellationToken);
        }

        public async Task PlaySequenceAsync(IEnumerable<string> texts, CancellationToken cancellationToken = default)
        {
            if (texts == null)
            {
                return;
            }

            foreach (var text in texts)
            {
                if (cancellationToken.IsCancellationRequested)
                {
                    return;
                }

                await PlayByTextAsync(text, cancellationToken);
            }
        }

        public Task<bool> PlayClipAsync(AudioClip clip, CancellationToken cancellationToken = default)
        {
            if (overlapPolicy == OverlapPolicy.Queue)
            {
                return EnqueuePlayback(() => PlayClipInternalAsync(clip, cancellationToken));
            }

            return PlayClipInternalAsync(clip, cancellationToken);
        }

        public void ClearCache()
        {
            _cache.Clear();
        }

        private async Task<bool> PlayByTextInternalAsync(string text, CancellationToken cancellationToken)
        {
            if (string.IsNullOrWhiteSpace(text))
            {
                return false;
            }

            var cacheKey = $"tts::{text.Trim()}";
            if (enableCaching && _cache.TryGetValue(cacheKey, out var cachedClip) && cachedClip != null)
            {
                return await PlayClipInternalAsync(cachedClip, cancellationToken);
            }

            var result = await _backendClient.RequestAudioFromTextAsync(text, cancellationToken);
            if (!result.IsSuccess || result.Clip == null)
            {
                Debug.LogWarning($"Failed to fetch TTS audio for text '{text}': {result.Error}", this);
                BubbleEventBus.PublishAudioPlaybackFailed(text);
                return false;
            }

            if (enableCaching)
            {
                _cache[cacheKey] = result.Clip;
            }

            return await PlayClipInternalAsync(result.Clip, cancellationToken);
        }

        private Task<bool> EnqueuePlayback(Func<Task<bool>> playbackOperation)
        {
            var completionSource = new TaskCompletionSource<bool>();

            lock (_queueLock)
            {
                _queueTail = _queueTail.ContinueWith(async _ =>
                {
                    try
                    {
                        var result = await playbackOperation();
                        completionSource.TrySetResult(result);
                    }
                    catch (Exception exception)
                    {
                        completionSource.TrySetException(exception);
                    }
                }).Unwrap();
            }

            return completionSource.Task;
        }

        private async Task<bool> PlayClipInternalAsync(AudioClip clip, CancellationToken cancellationToken)
        {
            if (clip == null || playbackSource == null)
            {
                return false;
            }

            if (playbackSource.isPlaying)
            {
                if (overlapPolicy == OverlapPolicy.SkipIfBusy)
                {
                    return false;
                }

                playbackSource.Stop();
            }

            playbackSource.clip = clip;
            playbackSource.Play();
            BubbleEventBus.PublishAudioPlaybackStarted(clip.name);

            while (playbackSource.isPlaying)
            {
                if (cancellationToken.IsCancellationRequested)
                {
                    playbackSource.Stop();
                    BubbleEventBus.PublishAudioPlaybackFailed(clip.name);
                    return false;
                }

                await Task.Yield();
            }

            BubbleEventBus.PublishAudioPlaybackEnded(clip.name);
            return true;
        }
    }
}
