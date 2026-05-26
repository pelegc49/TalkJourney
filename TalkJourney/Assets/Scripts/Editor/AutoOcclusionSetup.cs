using UnityEngine;
using UnityEditor;

public class AutoOcclusionSetup
{
    // הגדרת גודל מינימלי לאובייקט מסתיר (במטרים)
    private const float MIN_OCCLUDER_SIZE = 1.0f;
    // ערך המציין חומרים שקופים במערכת הרינדור
    private const int TRANSPARENT_QUEUE_THRESHOLD = 3000;

    [MenuItem("Tools/Auto Setup Occlusion Flags")]
    public static void SetupFlags()
    {
        GameObject[] selectedObjects = Selection.gameObjects;

        if (selectedObjects.Length == 0)
        {
            Debug.LogWarning("אנא סמן את האובייקטים בחלון ההיררכיה תחילה.");
            return;
        }

        int processedCount = 0;
        int occludersCount = 0;

        foreach (GameObject obj in selectedObjects)
        {
            Renderer renderer = obj.GetComponent<Renderer>();
            
            // אם אין לאובייקט מודל ויזואלי, נדלג עליו
            if (renderer == null || renderer.sharedMaterial == null)
            {
                continue;
            }

            processedCount++;

            // כברירת מחדל, מתחילים עם הדגלים הקיימים ומוסיפים להם את דגל המוסתר
            StaticEditorFlags currentFlags = GameObjectUtility.GetStaticEditorFlags(obj);
            currentFlags |= StaticEditorFlags.OccludeeStatic;

            bool isTransparent = renderer.sharedMaterial.renderQueue >= TRANSPARENT_QUEUE_THRESHOLD;
            bool isLargeEnough = renderer.bounds.size.magnitude >= MIN_OCCLUDER_SIZE;

            if (isTransparent || !isLargeEnough)
            {
                // מסירים את דגל המסתיר מאובייקטים קטנים או שקופים
                currentFlags &= ~StaticEditorFlags.OccluderStatic;
            }
            else
            {
                // מוסיפים את דגל המסתיר לאובייקטים גדולים ואטומים
                currentFlags |= StaticEditorFlags.OccluderStatic;
                occludersCount++;
            }

            // החלת הדגלים החדשים על האובייקט
            GameObjectUtility.SetStaticEditorFlags(obj, currentFlags);
        }

        Debug.Log($"התהליך הסתיים: {processedCount} אובייקטים עודכנו. מתוכם {occludersCount} הוגדרו גם כמסתירים.");
    }
}