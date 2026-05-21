using System;
using System.Threading;
using System.Threading.Tasks;
using TalkJourney.BubbleSystem.Audio;
using TalkJourney.BubbleSystem.Events;
using TalkJourney.BubbleSystem.Localization;
using TMPro;
using UnityEngine;

/// <summary>
/// Central guide controller (singleton) that plays localized subtitles and TTS audio.
/// Integrates with LocalizationResolver and AudioBackendClient from the project.
/// Manages voice-recognition failure tracking and emits a fallback-unlocked signal.
/// </summary>
public class GuideController : MonoBehaviour
{
    public static GuideController Instance { get; private set; }

    [Header("References")]
    [Tooltip("AudioSource used to play voice instructions. If null, will try to get one on this GameObject.")]
    public AudioSource audioSource;

    [Tooltip("TextMeshProUGUI used for subtitles (UI). Optional but recommended.")]
    public TextMeshProUGUI subtitleText;

    [Tooltip("Visual guide character. If assigned, will show/hide during playback with spatial audio.")]
    public VisualGuide visualGuide;

    [Header("Services (auto-resolve if left empty)")]
    public LocalizationResolver localizationResolver;
    public AudioBackendClient audioBackendClient;

    [Header("Behavior")]
    [Tooltip("How long (seconds) to wait before clearing the subtitle after playback finishes.")]
    public float subtitleClearDelay = 1.0f;

    [Header("Subtitle World Space Follow")]
    [Tooltip("When enabled, subtitle text transform follows a target in world space.")]
    public bool enableWorldSpaceSubtitleFollow = false;

    [Tooltip("Target transform to follow (typically XR camera/head). If empty, Camera.main is used when available.")]
    public Transform subtitleFollowTarget;

    [Tooltip("World-space offset from subtitleFollowTarget position.")]
    public Vector3 subtitleWorldOffset = new Vector3(0f, -0.2f, 1.2f);

    [Tooltip("Smooth follow speed. Set <= 0 for snap movement.")]
    public float subtitleFollowLerpSpeed = 10f;

    [Tooltip("If enabled, the subtitle faces the main camera.")]
    public bool subtitleFaceCamera = true;

    private CancellationTokenSource _playbackCts;

    /// <summary>
    /// Raised when voice recognition has failed enough times that the bypass path should unlock.
    /// The existing bypass system can subscribe to this and reveal/enable its own UI.
    /// </summary>
    public static event Action VoiceFallbackUnlocked;

    // Voice recognition failure tracking
    private int _consecutiveVoiceFails = 0;
    private const int MAX_FAILS = 3;

    void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);
            return;
        }
        Instance = this;

        if (audioSource == null)
            audioSource = GetComponent<AudioSource>();

        // Try to auto-resolve services if they were not assigned in inspector
        if (localizationResolver == null)
            localizationResolver = FindFirstObjectByType<LocalizationResolver>();

        if (audioBackendClient == null)
            audioBackendClient = FindFirstObjectByType<AudioBackendClient>();
    }

    void OnDestroy()
    {
        if (Instance == this) Instance = null;
        _playbackCts?.Cancel();
        _playbackCts?.Dispose();
    }

    void LateUpdate()
    {
        if (!enableWorldSpaceSubtitleFollow || subtitleText == null)
            return;

        var follow = subtitleFollowTarget;
        if (follow == null && Camera.main != null)
        {
            follow = Camera.main.transform;
        }

        if (follow == null)
            return;

        var targetPosition = follow.TransformPoint(subtitleWorldOffset);
        // Keep Y tied to world-height + offset (jump/crouch), not head pitch/roll rotation.
        targetPosition.y = follow.position.y + subtitleWorldOffset.y;
        var subtitleTransform = subtitleText.transform;

        if (subtitleFollowLerpSpeed <= 0f)
        {
            subtitleTransform.position = targetPosition;
        }
        else
        {
            subtitleTransform.position = Vector3.Lerp(
                subtitleTransform.position,
                targetPosition,
                Time.deltaTime * subtitleFollowLerpSpeed);
        }

        if (subtitleFaceCamera && Camera.main != null)
        {
            var cam = Camera.main.transform;
            var lookDirection = subtitleTransform.position - cam.position;
            if (lookDirection.sqrMagnitude > 0.0001f)
            {
                subtitleTransform.rotation = Quaternion.LookRotation(lookDirection.normalized, Vector3.up);
            }
        }
    }

    /// <summary>
    /// Play a guide instruction: show visual guide, fetch localized text and remote audio, update subtitle UI and play audio.
    /// Audio plays from the visual guide's position (spatial 3D audio).
    /// This method is safe to call multiple times; previous playback will be cancelled when a new call arrives.
    /// </summary>
    public async Task PlayInstruction(string dialogueKey)
    {
        await EndCurrentInstructionAsync();

        _playbackCts?.Dispose();
        _playbackCts = new CancellationTokenSource();
        var currentPlaybackCts = _playbackCts;
        var token = _playbackCts.Token;

        try
        {
            // Show visual guide
            if (visualGuide != null && Camera.main != null)
            {
                visualGuide.Show(Camera.main.transform);
            }

            // Resolve text
            string subtitle = dialogueKey;
            try
            {
                if (localizationResolver != null)
                    subtitle = localizationResolver.ResolveForLanguage(dialogueKey, localizationResolver.nativeLanguage) ?? dialogueKey;
            }
            catch (Exception) { /* swallow - fall back to key */ }

            if (subtitleText != null)
            {
                var useRtl = localizationResolver != null
                    && LocalizationResolver.IsRightToLeftLanguage(localizationResolver.nativeLanguage);
                subtitleText.isRightToLeftText = useRtl;
                subtitleText.alignment = useRtl ? TMPro.TextAlignmentOptions.Right : TMPro.TextAlignmentOptions.Left;
                subtitleText.text = subtitle;
                subtitleText.gameObject.SetActive(true);
            }

            // Fetch audio from backend (may be null/throw)
            AudioClip clip = null;
            try
            {
                if (audioBackendClient != null)
                {
                    var audioResult = await audioBackendClient.RequestAudioFromTextAsync(subtitle, localizationResolver?.nativeLanguage, token);
                    if (audioResult.IsSuccess)
                    {
                        clip = audioResult.Clip;
                    }
                    else
                    {
                        Debug.LogWarning($"GuideController: audio request failed for '{dialogueKey}': {audioResult.Error}");
                    }
                }
            }
            catch (Exception e)
            {
                Debug.LogWarning($"GuideController: audio fetch failed for '{dialogueKey}': {e.Message}");
            }

            // Play audio if available
            if (clip != null)
            {
                // Use visual guide's audio source for spatial 3D audio, otherwise use fallback
                AudioSource effectiveAudioSource = visualGuide != null ? visualGuide.audioSource : audioSource;

                if (effectiveAudioSource != null)
                {
                    if (visualGuide != null)
                    {
                        visualGuide.StartTalking(clip);
                    }
                    else
                    {
                        effectiveAudioSource.PlayOneShot(clip);
                    }

                    try
                    {
                        await Task.Delay(TimeSpan.FromSeconds(clip.length), token);
                    }
                    catch (TaskCanceledException)
                    {
                        return;
                    }
                }
            }
            else
            {
                // If no audio, leave subtitle a short while so player can read
                try { await Task.Delay(TimeSpan.FromSeconds(1.5f), token); }
                catch (TaskCanceledException)
                {
                    return;
                }
            }

            if (token.IsCancellationRequested)
            {
                return;
            }

            // Clear subtitle after a short delay
            if (subtitleText != null)
            {
                try { await Task.Delay(TimeSpan.FromSeconds(subtitleClearDelay), token); }
                catch (TaskCanceledException)
                {
                    return;
                }

                if (!token.IsCancellationRequested)
                    subtitleText.text = string.Empty;
            }

            if (token.IsCancellationRequested)
            {
                return;
            }

            // Hide visual guide with disappear animation
            if (visualGuide != null)
            {
                visualGuide.StopTalking();
                await visualGuide.HideAsync();
            }
        }
        catch (Exception ex)
        {
            Debug.LogException(ex);
        }
        finally
        {
            if (_playbackCts == currentPlaybackCts)
            {
                _playbackCts = null;
            }
        }
    }

    /// <summary>
    /// Stops the current guide line immediately and waits for the visual guide to finish disappearing.
    /// Safe to call even when no guide line is active.
    /// </summary>
    public async Task EndCurrentInstructionAsync()
    {
        var activePlaybackCts = _playbackCts;
        if (activePlaybackCts != null)
        {
            activePlaybackCts.Cancel();
        }

        if (audioSource != null)
        {
            audioSource.Stop();
        }

        if (visualGuide != null)
        {
            visualGuide.StopTalking();
        }

        if (subtitleText != null)
        {
            subtitleText.text = string.Empty;
        }

        if (visualGuide != null)
        {
            await visualGuide.HideAsync();
        }

        if (_playbackCts == activePlaybackCts)
        {
            _playbackCts = null;
        }
    }

    /// <summary>
    /// Called by external voice-recognition layer to report a recognition attempt result.
    /// If repeated failures occur, prompts player to use controller and emits the bypass-unlock signal.
    /// </summary>
    public void RegisterVoiceAttempt(bool isRecognized)
    {
        if (isRecognized)
        {
            _consecutiveVoiceFails = 0;
            return;
        }

        _consecutiveVoiceFails++;

        if (_consecutiveVoiceFails < MAX_FAILS)
        {
            // Ask the player to try again (short hint)
            _ = PlayInstruction("guide_try_again_key");
        }
        else
        {
            // Fail-safe: after reaching max fails, tell player to use controller and unlock bypass.
            _consecutiveVoiceFails = 0;
            _ = PlayInstruction("guide_use_controller_key");
            VoiceFallbackUnlocked?.Invoke();
        }
    }

    #region Editor helpers
    [ContextMenu("Simulate Voice Fail")]
    private void SimulateVoiceFail()
    {
        RegisterVoiceAttempt(false);
    }

    [ContextMenu("Simulate Voice Success")]
    private void SimulateVoiceSuccess()
    {
        RegisterVoiceAttempt(true);
    }
    #endregion
}
