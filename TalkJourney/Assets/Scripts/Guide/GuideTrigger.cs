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
    public string playerTag = "Player";

    private bool _hasTriggered = false;

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

        if (!other.CompareTag(playerTag))
            return;

        if (GuideController.Instance != null)
        {
            _ = GuideController.Instance.PlayInstruction(dialogueKey);
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
