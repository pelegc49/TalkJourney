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

    [Tooltip("Optional GuidePointer instance used to visualize targets in the world.")]
    public GuidePointer guidePointer;

    [Header("Services (auto-resolve if left empty)")]
    public LocalizationResolver localizationResolver;
    public AudioBackendClient audioBackendClient;

    [Header("Behavior")]
    [Tooltip("How long (seconds) to wait before clearing the subtitle after playback finishes.")]
    public float subtitleClearDelay = 1.0f;

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

    /// <summary>
    /// Play a guide instruction: fetch localized text and remote audio, update subtitle UI and play audio.
    /// This method is safe to call multiple times; previous playback will be cancelled when a new call arrives.
    /// </summary>
    public async Task PlayInstruction(string dialogueKey)
    {
        _playbackCts?.Cancel();
        _playbackCts?.Dispose();
        _playbackCts = new CancellationTokenSource();
        var token = _playbackCts.Token;

        try
        {
            // Resolve text
            string subtitle = dialogueKey;
            try
            {
                if (localizationResolver != null)
                    subtitle = localizationResolver.Resolve(dialogueKey) ?? dialogueKey;
            }
            catch (Exception) { /* swallow - fall back to key */ }

            if (subtitleText != null)
            {
                subtitleText.text = subtitle;
                subtitleText.gameObject.SetActive(true);
            }

            // Fetch audio from backend (may be null/throw)
            AudioClip clip = null;
            try
            {
                if (audioBackendClient != null)
                {
                    var audioResult = await audioBackendClient.RequestAudioFromTextAsync(subtitle, token);
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
            if (clip != null && audioSource != null)
            {
                audioSource.PlayOneShot(clip);
                try
                {
                    await Task.Delay(TimeSpan.FromSeconds(clip.length), token);
                }
                catch (TaskCanceledException) { }
            }
            else
            {
                // If no audio, leave subtitle a short while so player can read
                try { await Task.Delay(TimeSpan.FromSeconds(1.5f), token); } catch (TaskCanceledException) { }
            }

            // Clear subtitle after a short delay
            if (subtitleText != null)
            {
                try { await Task.Delay(TimeSpan.FromSeconds(subtitleClearDelay), token); } catch (TaskCanceledException) { }
                if (!token.IsCancellationRequested)
                    subtitleText.text = string.Empty;
            }
        }
        catch (Exception ex)
        {
            Debug.LogException(ex);
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

    /// <summary>
    /// Show the world-space pointer above the specified target Transform.
    /// Call with null to hide.
    /// </summary>
    public void ShowPointer(Transform target)
    {
        if (guidePointer == null)
            return;

        if (target == null)
            guidePointer.Clear();
        else
            guidePointer.PointAt(target);
    }

    /// <summary>
    /// Hide any active guide pointer.
    /// </summary>
    public void HidePointer()
    {
        if (guidePointer == null)
            return;

        guidePointer.Clear();
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
