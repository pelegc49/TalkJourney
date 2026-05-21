using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.Events;

/// <summary>
/// World-space arrow that points at a target Transform.
/// Positions itself above the target, faces the player camera, and adds a subtle bob/scale pulse.
/// </summary>
public class GuidePointer : MonoBehaviour
{
    [Tooltip("Offset from target position where the arrow will be placed (X/Y/Z in world space).")]
    public Vector3 targetOffset = new Vector3(0f, 0.6f, 0f);

    [Tooltip("Additional Euler rotation offset (X/Y/Z) applied after camera-facing rotation.")]
    public Vector3 rotationOffsetEuler = Vector3.zero;

    [Tooltip("Bob amplitude in meters.")]
    public float bobAmplitude = 0.05f;

    [Tooltip("Bob speed in cycles per second.")]
    public float bobSpeed = 1.2f;

    [Tooltip("Pulse scale amount.")]
    public float pulseAmount = 0.08f;

    [Tooltip("Pulse speed in cycles per second.")]
    public float pulseSpeed = 1.2f;

    [Header("Input")]
    [Tooltip("Optional: Assign the XRI Default Input Actions Activate action to activate pointer via VR controller button.")]
    public InputActionReference activateAction;

    /// <summary>
    /// Event fired when the pointer is activated via trigger button while active.
    /// </summary>
    public UnityEvent OnPointerActivated;

    private Transform _target;
    private Vector3 _basePosition;
    private Vector3 _baseScale;
    private Renderer[] _childRenderers;
    private bool _active = false;

    void Awake()
    {
        _baseScale = transform.localScale;
        _childRenderers = GetComponentsInChildren<Renderer>(true);
    }

    void OnEnable()
    {
        _active = true;
        if (activateAction != null && activateAction.action != null)
        {
            activateAction.action.performed += OnActivatePerformed;
        }
    }

    void OnDisable()
    {
        _active = false;
        if (activateAction != null && activateAction.action != null)
        {
            activateAction.action.performed -= OnActivatePerformed;
        }
    }

    private void OnActivatePerformed(InputAction.CallbackContext context)
    {
        if (_active && _target != null)
        {
            OnPointerActivated?.Invoke();
        }
    }

    void Update()
    {
        if (!_active || _target == null)
            return;

        var cam = Camera.main;
        // position above target
        _basePosition = _target.position + targetOffset;

        // Compute target rotation first so bob can follow the pointer's rotated up axis (including Z roll offset).
        Quaternion targetRotation;
        if (cam != null)
        {
            var lookDir = (cam.transform.position - _basePosition).normalized;
            targetRotation = Quaternion.LookRotation(lookDir, Vector3.up) * Quaternion.Euler(rotationOffsetEuler);
        }
        else
        {
            targetRotation = Quaternion.Euler(rotationOffsetEuler);
        }

        // bob
        var bob = Mathf.Sin(Time.time * bobSpeed * Mathf.PI * 2f) * bobAmplitude;
        var bobAxis = targetRotation * Vector3.up;
        transform.position = _basePosition + bobAxis * bob;
        transform.rotation = targetRotation;

        // pulse scale
        var pulse = 1.0f + Mathf.Sin(Time.time * pulseSpeed * Mathf.PI * 2f) * pulseAmount;
        transform.localScale = _baseScale * pulse;
    }

    /// <summary>
    /// Start pointing at the given target transform.
    /// </summary>
    public void PointAt(Transform target)
    {
        _target = target;
        if (_target == null)
        {
            gameObject.SetActive(false);
            return;
        }

        gameObject.SetActive(true);
        SetRenderersEnabled(true);
        _basePosition = _target.position + targetOffset;
    }

    /// <summary>
    /// Stop pointing and hide the arrow.
    /// </summary>
    public void Clear()
    {
        _target = null;
        SetRenderersEnabled(false);
        gameObject.SetActive(false);
    }

    private void SetRenderersEnabled(bool isEnabled)
    {
        if (_childRenderers == null)
            return;

        for (int i = 0; i < _childRenderers.Length; i++)
        {
            var renderer = _childRenderers[i];
            if (renderer != null)
            {
                renderer.enabled = isEnabled;
            }
        }
    }
}
