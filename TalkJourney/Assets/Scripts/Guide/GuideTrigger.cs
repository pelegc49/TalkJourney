using UnityEngine;

/// <summary>
/// Simple trigger component to invoke guide lines when the player enters a trigger volume.
/// Attach to a GameObject with a trigger BoxCollider. Uses the "Player" tag by default.
/// </summary>
[RequireComponent(typeof(Collider))]
public class GuideTrigger : MonoBehaviour
{
    [Tooltip("Localization/audio key that the GuideController will play when triggered.")]
    public string dialogueKey = "";

    [Tooltip("If true, this trigger will only fire once and then ignore further entries.")]
    public bool triggerOnce = true;

    [Tooltip("Tag used to identify the player collider. Defaults to 'Player'.")]
    public string playerTag = "MainCamera";

    private bool _hasTriggered = false;

    [Header("Pointer")]
    [Tooltip("Guide pointer instance owned by this trigger.")]
    public GuidePointer guidePointer;

    [Tooltip("Optional target Transform that the guide arrow will point at when triggered.")]
    public Transform pointerTarget;

    [Tooltip("If true, show the guide pointer on trigger enter when a pointer target is set.")]
    public bool showPointerOnEnter = true;

    void Reset()
    {
        // Ensure collider is a trigger by default when added
        var col = GetComponent<Collider>();
        if (col != null)
            col.isTrigger = true;
    }

    void OnTriggerEnter(Collider other)
    {
        if (_hasTriggered && triggerOnce)
            return;

        if (!HasPlayerTagInHierarchy(other))
            return;

        ExecuteGuide();
    }

    private bool HasPlayerTagInHierarchy(Collider other)
    {
        if (other == null)
            return false;

        if (other.CompareTag(playerTag))
            return true;

        var current = other.transform;
        while (current != null)
        {
            if (current.CompareTag(playerTag))
                return true;

            current = current.parent;
        }

        foreach (Transform child in other.transform)
        {
            if (child.CompareTag(playerTag))
                return true;
        }

        return false;
    }

    /// <summary>
    /// Public entry point for Unity UI Button onClick.
    /// </summary>
    public void TriggerFromUI()
    {
        ExecuteGuide();
    }

    private void ExecuteGuide()
    {
        if (_hasTriggered && triggerOnce)
            return;

        if (GuideController.Instance != null)
        {
            _ = GuideController.Instance.PlayInstruction(dialogueKey);
            if (showPointerOnEnter && guidePointer != null && pointerTarget != null)
            {
                guidePointer.PointAt(pointerTarget);
            }
        }
        else
        {
            Debug.LogWarning("GuideTrigger: GuideController.Instance not found.");
        }

        if (triggerOnce)
            _hasTriggered = true;
    }

    [ContextMenu("Reset Trigger")]
    public void ResetTrigger()
    {
        _hasTriggered = false;
    }
}
