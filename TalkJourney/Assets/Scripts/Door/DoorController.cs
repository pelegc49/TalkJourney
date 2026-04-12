using UnityEngine;

public class DoorController : MonoBehaviour
{
    private Animator animator;
    private bool isDoorOpen = false;
    
    [Tooltip("גרור לכאן את בועת ההנחיה שיצרת")]
    public GameObject instructionBubble;

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
    }

    // פונקציה לפתיחה וסגירה של הדלת
    public void ToggleDoor()
    {
        isDoorOpen = !isDoorOpen; // הופך את המצב (אם פתוח אז נסגר, ולהפך)
        animator.SetBool("IsOpen", isDoorOpen);
    }

    // פונקציות להצגה והסתרה של הבועה
    public void ShowBubble()
    {
        if (instructionBubble != null)
        {
            instructionBubble.SetActive(true);
        }
    }

    public void HideBubble()
    {
        if (instructionBubble != null)
        {
            instructionBubble.SetActive(false);
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