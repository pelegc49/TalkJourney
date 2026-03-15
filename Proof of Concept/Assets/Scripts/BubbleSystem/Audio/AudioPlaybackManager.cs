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

        public Task<bool> PlayByIdentifierAsync(string audioIdentifier, CancellationToken cancellationToken = default)
        {
            if (_backendClient == null)
            {
                return Task.FromResult(false);
            }

            if (overlapPolicy == OverlapPolicy.Queue)
            {
                return EnqueuePlayback(() => PlayByIdentifierInternalAsync(audioIdentifier, cancellationToken));
            }

            return PlayByIdentifierInternalAsync(audioIdentifier, cancellationToken);
        }

        public async Task PlaySequenceAsync(IEnumerable<string> audioIdentifiers, CancellationToken cancellationToken = default)
        {
            if (audioIdentifiers == null)
            {
                return;
            }

            foreach (var audioIdentifier in audioIdentifiers)
            {
                if (cancellationToken.IsCancellationRequested)
                {
                    return;
                }

                await PlayByIdentifierAsync(audioIdentifier, cancellationToken);
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

        private async Task<bool> PlayByIdentifierInternalAsync(string audioIdentifier, CancellationToken cancellationToken)
        {
            if (string.IsNullOrWhiteSpace(audioIdentifier))
            {
                return false;
            }

            var clip = await GetClipAsync(audioIdentifier.Trim(), cancellationToken);
            if (clip == null)
            {
                return false;
            }

            return await PlayClipInternalAsync(clip, cancellationToken);
        }

        private async Task<AudioClip> GetClipAsync(string audioIdentifier, CancellationToken cancellationToken)
        {
            if (enableCaching && _cache.TryGetValue(audioIdentifier, out var cachedClip) && cachedClip != null)
            {
                return cachedClip;
            }

            var result = await _backendClient.RequestAudioAsync(audioIdentifier, cancellationToken);
            if (!result.IsSuccess || result.Clip == null)
            {
                Debug.LogWarning($"Failed to fetch audio '{audioIdentifier}': {result.Error}", this);
                BubbleEventBus.PublishAudioPlaybackFailed(audioIdentifier);
                return null;
            }

            if (enableCaching)
            {
                _cache[audioIdentifier] = result.Clip;
            }

            return result.Clip;
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
