using UnityEngine;

/// <summary>
/// Simple character customizer that uses direct prefab references.
/// Attach this script to your character prefab instance and assign attachments in the Inspector.
/// </summary>
public class CharacterCustomizerSimple : MonoBehaviour
{
    [System.Serializable]
    public class AttachmentSlot
    {
        public GameObject attachmentPrefab;  // Assign via Inspector
        public string targetBoneName;         // e.g., "Head", "Armature", "Skeleton"
        public Vector3 positionOffset;
        public Vector3 rotationOffset;
    }

    [SerializeField] private AttachmentSlot[] attachments = new AttachmentSlot[0];
    private Transform[] attachmentInstances;

    void Start()
    {
        ApplyAttachments();
    }

    /// <summary>
    /// Applies all configured attachments to the character
    /// </summary>
    public void ApplyAttachments()
    {
        RemoveAttachments();
        attachmentInstances = new Transform[attachments.Length];

        for (int i = 0; i < attachments.Length; i++)
        {
            if (attachments[i].attachmentPrefab == null)
            {
                Debug.LogWarning($"Attachment slot {i} is empty");
                continue;
            }

            Transform targetBone = FindBoneRecursive(transform, attachments[i].targetBoneName);
            if (targetBone == null)
            {
                Debug.LogWarning($"Could not find bone '{attachments[i].targetBoneName}'");
                continue;
            }

            GameObject instance = Instantiate(attachments[i].attachmentPrefab, targetBone);
            instance.name = attachments[i].attachmentPrefab.name;
            instance.transform.localPosition = attachments[i].positionOffset;
            instance.transform.localRotation = Quaternion.Euler(attachments[i].rotationOffset);

            attachmentInstances[i] = instance.transform;
        }
    }

    /// <summary>
    /// Removes all attachments
    /// </summary>
    public void RemoveAttachments()
    {
        if (attachmentInstances == null) return;

        foreach (Transform attachment in attachmentInstances)
        {
            if (attachment != null)
                Destroy(attachment.gameObject);
        }
        attachmentInstances = null;
    }

    /// <summary>
    /// Finds a bone by name recursively in the character hierarchy
    /// </summary>
    private Transform FindBoneRecursive(Transform current, string boneName)
    {
        if (current.name == boneName)
            return current;

        for (int i = 0; i < current.childCount; i++)
        {
            Transform result = FindBoneRecursive(current.GetChild(i), boneName);
            if (result != null)
                return result;
        }

        return null;
    }

    /// <summary>
    /// Logs all bone names in the hierarchy - use this to find the right bone names!
    /// </summary>
    public void LogAllBones()
    {
        Debug.Log("=== All Bones in Character Hierarchy ===");
        LogBonesRecursive(transform, 0);
    }

    private void LogBonesRecursive(Transform current, int depth)
    {
        string indent = new string(' ', depth * 2);
        Debug.Log($"{indent}{current.name}");

        for (int i = 0; i < current.childCount; i++)
        {
            LogBonesRecursive(current.GetChild(i), depth + 1);
        }
    }
}
