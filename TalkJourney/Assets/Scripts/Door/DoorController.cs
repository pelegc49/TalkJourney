using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.XR.Interaction.Toolkit;
using UnityEngine.XR.Interaction.Toolkit.Interactables;

[RequireComponent(typeof(XRSimpleInteractable))]
public class DoorController : MonoBehaviour
{
    private Animator animator;
    private bool isDoorOpen = false;
    private XRSimpleInteractable xrSimpleInteractable;
    
    [Header("Input")]
    [Tooltip("Assign the XRI Default Input Actions Activate action from the XRI Right Interaction map.")]
    public InputActionReference activateAction;

    [Tooltip("Drag the instruction bubble you created here")]
    public GameObject instructionBubble;
    private Animator instructionBubbleAnimator;
    private bool isDoorHover = false;

    private void Awake()
    {
        xrSimpleInteractable = GetComponent<XRSimpleInteractable>();

        if (instructionBubble == null)
        {
            Transform bubble = FindDirectChild(transform, "InstructionBubble");
            if (bubble != null)
            {
                instructionBubble = bubble.gameObject;
            }
        }

        if (xrSimpleInteractable != null)
        {
            xrSimpleInteractable.hoverEntered.AddListener(OnHoverEntered);
            xrSimpleInteractable.hoverExited.AddListener(OnHoverExited);
        }
    }

    private void Start()
    {
        animator = GetComponent<Animator>();
        if (instructionBubble != null)
        {
            instructionBubbleAnimator = instructionBubble.GetComponent<Animator>();
        }
    }

    private void OnEnable()
    {
        if (activateAction != null && activateAction.action != null)
        {
            activateAction.action.performed += OnActivatePerformed;
        }
    }

    private void OnDisable()
    {
        if (activateAction != null && activateAction.action != null)
        {
            activateAction.action.performed -= OnActivatePerformed;
        }

        if (xrSimpleInteractable != null)
        {
            xrSimpleInteractable.hoverEntered.RemoveListener(OnHoverEntered);
            xrSimpleInteractable.hoverExited.RemoveListener(OnHoverExited);
        }
    }

    private void OnActivatePerformed(InputAction.CallbackContext context)
    {
        if (!isDoorHover)
        {
            return;
        }

        ToggleDoor();
    }

    private void OnHoverEntered(HoverEnterEventArgs args)
    {
        ShowBubble();
    }

    private void OnHoverExited(HoverExitEventArgs args)
    {
        HideBubble();
    }

    public void ToggleDoor()
    {
        isDoorOpen = !isDoorOpen;

        if (animator != null)
        {
            animator.SetBool("IsOpen", isDoorOpen);
        }
    }

    public void ShowBubble()
    {
        isDoorHover = true;

        if (instructionBubbleAnimator != null)
        {
            instructionBubbleAnimator.SetBool("IsHover", isDoorHover);
        }
    }

    public void HideBubble()
    {
        isDoorHover = false;

        if (instructionBubbleAnimator != null)
        {
            instructionBubbleAnimator.SetBool("IsHover", isDoorHover);
        }
    }

    private Transform FindDirectChild(Transform parent, string childName)
    {
        for (int i = 0; i < parent.childCount; i++)
        {
            Transform child = parent.GetChild(i);
            if (child.name == childName)
            {
                return child;
            }
        }

        return null;
    }
}