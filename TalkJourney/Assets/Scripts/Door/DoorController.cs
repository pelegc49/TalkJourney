using UnityEngine;

public class DoorController : MonoBehaviour
{
    private Animator animator;
    private bool isDoorOpen = false;
    
    [Tooltip("Drag the instruction bubble you created here")]
    public GameObject instructionBubble;
    private Animator instructionBubbleAnimator;
    private bool isDoorHover = false;

    private void Awake()
    {
        if (instructionBubble == null)
        {
            Transform bubble = FindDirectChild(transform, "InstructionBubble");
            if (bubble != null)
            {
                instructionBubble = bubble.gameObject;
            }
        }
    }

    void Start()
    {
        animator = GetComponent<Animator>();
        instructionBubbleAnimator = instructionBubble.GetComponent<Animator>();
    }

    // Function to open and close the door
    public void ToggleDoor()
    {
        isDoorOpen = !isDoorOpen; // Toggle state (if open then close, and vice versa)
        animator.SetBool("IsOpen", isDoorOpen);
    }

    // Functions to show and hide the bubble
    public void ShowBubble()
    {
        isDoorHover = true;
        instructionBubbleAnimator.SetBool("IsHover", isDoorHover);
    }

    public void HideBubble()
    {
        isDoorHover = false;
        instructionBubbleAnimator.SetBool("IsHover", isDoorHover);
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