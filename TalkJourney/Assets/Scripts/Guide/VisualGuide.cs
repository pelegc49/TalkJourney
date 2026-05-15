using System.Threading.Tasks;
using UnityEngine;

/// <summary>
/// Visual guide character that appears when guide instructions play.
/// Moves into player's view, looks at player while talking, and disappears when done.
/// Audio is played from the guide's position (spatial 3D audio).
/// </summary>
[RequireComponent(typeof(Animator))]
[RequireComponent(typeof(AudioSource))]
public class VisualGuide : MonoBehaviour
{
    [Header("References")]
    [Tooltip("Animator component for guide animations")]
    public Animator animator;

    [Tooltip("AudioSource for spatial audio from guide position")]
    public AudioSource audioSource;

    [Header("Appearance")]
    [Tooltip("Time (seconds) for appear animation")]
    public float appearDuration = 1f;

    [Tooltip("Time (seconds) for disappear animation")]
    public float disappearDuration = 1f;

    [Header("Movement")]
    [Tooltip("Speed at which guide moves into player view")]
    public float moveSpeed = 3f;

    [Tooltip("Distance from player to stop at when moving into view")]
    public float viewDistance = 2f;

    [Tooltip("Angle (degrees) - guide is considered 'in view' if within this cone")]
    public float viewAngleThreshold = 30f;

    [Header("Animation Parameters")]
    [Tooltip("Animator parameter name for visibility state (IdleOff -> Appear -> Idle -> Disappear -> IdleOff)")]
    public string isVisibleParameter = "IsVisible";

    [Tooltip("Animator parameter name for talking state")]
    public string talkingParameter = "IsTalking";

    [Header("Look")]
    [Tooltip("Yaw offset applied when looking at the player (degrees). Use 180 if model faces backwards.)")]
    public float lookYawOffsetDegrees = 180f;
    private Transform _playerCamera;
    private bool _isVisible = false;
    private bool _isTalking = false;
    private bool _isHiding = false;
    private Vector3 _initialPosition;
    private Quaternion _initialRotation;

    private void Awake()
    {
        if (animator == null)
            animator = GetComponent<Animator>();

        if (audioSource == null)
            audioSource = GetComponent<AudioSource>();

        _initialPosition = transform.position;
        _initialRotation = transform.rotation;

        // Ensure audio source is set up for 3D spatial audio
        audioSource.spatialBlend = 1f; // Fully 3D
        audioSource.dopplerLevel = 1f;
        audioSource.rolloffMode = AudioRolloffMode.Logarithmic;
        audioSource.playOnAwake = false;
        audioSource.Stop();
    }

    /// <summary>
    /// Show the visual guide with appear animation.
    /// </summary>
    public void Show(Transform playerCamera)
    {
        if (_isVisible)
            return;

        _playerCamera = playerCamera;
        _isVisible = true;
        _isHiding = false;
        gameObject.SetActive(true);

        // Reset position to initial or near player
        if (_playerCamera != null)
        {
            transform.position = _playerCamera.position + _playerCamera.forward * viewDistance;
        }
        else
        {
            transform.position = _initialPosition;
        }

        // Trigger appear animation
        animator.SetBool(isVisibleParameter, true);
    }

    /// <summary>
    /// Hide the visual guide with disappear animation, then deactivate.
    /// </summary>
    public void Hide()
    {
        _ = HideAsync();
    }

    /// <summary>
    /// Hide the visual guide with disappear animation, then deactivate.
    /// Returns when the hide animation duration has elapsed and the object is deactivated.
    /// </summary>
    public async Task HideAsync()
    {
        if (_isHiding)
            return;

        if (!_isVisible)
            return;

        _isHiding = true;
        _isVisible = false;
        _isTalking = false;
        animator.SetBool(talkingParameter, false);

        // Trigger disappear animation
        animator.SetBool(isVisibleParameter, false);

        // Deactivate after disappear animation completes
        // Keep GameObject active until coroutine finishes
        await Task.Delay((int)(disappearDuration * 1000f));
        if (gameObject != null && gameObject.activeInHierarchy)
        {
            gameObject.SetActive(false);
        }

        _isHiding = false;
    }

    /// <summary>
    /// Start playing audio and talking animation.
    /// </summary>
    public void StartTalking(AudioClip clip)
    {
        if (!_isVisible)
            return;

        _isTalking = true;
        animator.SetBool(talkingParameter, true);

        if (clip != null && audioSource != null)
        {
            // Stop any existing audio, reset position, then play the new clip
            try
            {
                audioSource.Stop();
                audioSource.clip = clip;
                audioSource.time = 0f;
                audioSource.Play();
            }
            catch (System.Exception) { audioSource.Play(); }
        }
    }

    /// <summary>
    /// Stop talking animation and audio.
    /// </summary>
    public void StopTalking()
    {
        _isTalking = false;
        animator.SetBool(talkingParameter, false);

        if (audioSource != null)
        {
            audioSource.Stop();
        }
    }

    /// <summary>
    /// Check if audio is still playing.
    /// </summary>
    public bool IsAudioPlaying => audioSource != null && audioSource.isPlaying;

    private void Update()
    {
        if (!_isVisible || _playerCamera == null)
            return;

        // Look at player
        LookAtPlayer();

        // Move to stay in player's view
        if (!IsInPlayerView())
        {
            MoveToPlayerView();
        }
    }

    /// <summary>
    /// Make guide face the player.
    /// </summary>
    private void LookAtPlayer()
    {
        Vector3 dirToPlayer = _playerCamera.position - transform.position;
        if (dirToPlayer.sqrMagnitude > 0.01f)
        {
            Quaternion targetRotation = Quaternion.LookRotation(dirToPlayer, Vector3.up) * Quaternion.Euler(0f, lookYawOffsetDegrees, 0f);
            transform.rotation = Quaternion.Lerp(transform.rotation, targetRotation, Time.deltaTime * 5f);
        }
    }

    /// <summary>
    /// Check if guide is within player's view cone.
    /// </summary>
    private bool IsInPlayerView()
    {
        Vector3 dirToGuide = (transform.position - _playerCamera.position).normalized;
        float angle = Vector3.Angle(_playerCamera.forward, dirToGuide);
        return angle <= viewAngleThreshold;
    }

    /// <summary>
    /// Move guide towards center of player's view.
    /// </summary>
    private void MoveToPlayerView()
    {
        Vector3 targetViewPos = _playerCamera.position + _playerCamera.forward * viewDistance;
        transform.position = Vector3.MoveTowards(
            transform.position,
            targetViewPos,
            moveSpeed * Time.deltaTime
        );
    }
}
