using UnityEngine;

/// <summary>
/// World-space arrow that points at a target Transform.
/// Positions itself above the target, faces the player camera, and adds a subtle bob/scale pulse.
/// </summary>
public class GuidePointer : MonoBehaviour
{
    [Tooltip("Vertical offset from the target's position where the arrow will be placed.")]
    public float heightOffset = 0.6f;

    [Tooltip("Bob amplitude in meters.")]
    public float bobAmplitude = 0.05f;

    [Tooltip("Bob speed in cycles per second.")]
    public float bobSpeed = 1.2f;

    [Tooltip("Pulse scale amount.")]
    public float pulseAmount = 0.08f;

    [Tooltip("Pulse speed in cycles per second.")]
    public float pulseSpeed = 1.2f;

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
    }

    void OnDisable()
    {
        _active = false;
    }

    void Update()
    {
        if (!_active || _target == null)
            return;

        var cam = Camera.main;
        // position above target
        _basePosition = _target.position + Vector3.up * heightOffset;

        // bob
        var bob = Mathf.Sin(Time.time * bobSpeed * Mathf.PI * 2f) * bobAmplitude;
        transform.position = _basePosition + Vector3.up * bob;

        // face camera (billboard)
        if (cam != null)
        {
            var lookDir = (cam.transform.position - transform.position).normalized;
            // keep arrow upright and face camera
            transform.rotation = Quaternion.LookRotation(lookDir, Vector3.up);
        }

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
        _basePosition = _target.position + Vector3.up * heightOffset;
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
