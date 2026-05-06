using UnityEngine;
using TalkJourney.BubbleSystem.Interaction;

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

    [Header("Pointer")]
    [Tooltip("Optional target Transform that the guide arrow will point at when triggered.")]
    public Transform pointerTarget;

    [Tooltip("If true, show the guide pointer on trigger enter when a pointer target is set.")]
    public bool showPointerOnEnter = true;

    [Header("Completion")]
    [Tooltip("Optional interactable on the target object. When clicked, the guide pointer will hide.")]
    public VrPointerInteractable completionInteractable;

    [Tooltip("If true, try to find a VrPointerInteractable on pointerTarget automatically.")]
    public bool autoResolveCompletionInteractable = true;

    void Reset()
    {
        // Ensure collider is a trigger by default when added
        var col = GetComponent<Collider>();
        if (col != null)
            col.isTrigger = true;
    }

    void OnEnable()
    {
        ResolveCompletionInteractable();
        if (completionInteractable != null)
        {
            completionInteractable.Clicked += OnCompletionClicked;
        }
    }

    void OnDisable()
    {
        if (completionInteractable != null)
        {
            completionInteractable.Clicked -= OnCompletionClicked;
        }
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
            if (showPointerOnEnter && pointerTarget != null)
            {
                GuideController.Instance.ShowPointer(pointerTarget);
            }
        }
        else
        {
            Debug.LogWarning("GuideTrigger: GuideController.Instance not found.");
        }

        if (triggerOnce)
            _hasTriggered = true;
    }

    private void OnCompletionClicked()
    {
        if (GuideController.Instance != null)
        {
            GuideController.Instance.HidePointer();
        }
    }

    private void ResolveCompletionInteractable()
    {
        if (!autoResolveCompletionInteractable || completionInteractable != null || pointerTarget == null)
        {
            return;
        }

        completionInteractable = pointerTarget.GetComponent<VrPointerInteractable>();
    }

    [ContextMenu("Reset Trigger")]
    public void ResetTrigger()
    {
        _hasTriggered = false;
    }
}
