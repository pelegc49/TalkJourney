using UnityEngine;
using UnityEditor;
using UnityEngine.SceneManagement;

public class SceneScanner
{
    [MenuItem("Tools/List Scene Hierarchy and Scripts")]
    public static void ListHierarchyAndScripts()
    {
        Debug.Log("<b>=== Starting Scene Hierarchy Scan ===</b>");
        
        // שליפת כל האובייקטים הראשיים (שאין להם הורה) בסצנה הנוכחית
        GameObject[] rootObjects = SceneManager.GetActiveScene().GetRootGameObjects();

        // מעבר על כל אובייקט ראשי והפעלת סריקה עליו ועל ילדיו
        foreach (GameObject rootObj in rootObjects)
        {
            ScanObjectAndChildren(rootObj, 0);
        }
        
        Debug.Log("<b>=== Scan Complete ===</b>");
    }

    // פונקציה רקורסיבית שסורקת אובייקט ואז קוראת לעצמה עבור כל אחד מהילדים שלו
    private static void ScanObjectAndChildren(GameObject obj, int depth)
    {
        // יצירת מרווח חזותי (חצים) לפי עומק האובייקט כדי שייראה כמו עץ
        string indent = new string('-', depth * 2) + (depth > 0 ? "> " : "");

        // שליפת כל הרכיבים
        Component[] components = obj.GetComponents<Component>();
        string componentNames = "";

        foreach (Component comp in components)
        {
            if (comp != null)
            {
                componentNames += comp.GetType().Name + ", ";
            }
            else
            {
                componentNames += "[Missing Script], ";
            }
        }

        // הסרת הפסיק והרווח האחרונים
        if (componentNames.Length > 0)
        {
            componentNames = componentNames.TrimEnd(',', ' ');
        }

        // הדפסת האובייקט הנוכחי עם ההזחה המתאימה
        Debug.Log($"{indent}<b>{obj.name}</b> | Scripts: {componentNames}", obj);

        // מעבר על כל הילדים של האובייקט הנוכחי (באמצעות ה Transform שלו)
        foreach (Transform child in obj.transform)
        {
            // קריאה חוזרת לפונקציה עבור הילד, עם עומק גדול ב-1
            ScanObjectAndChildren(child.gameObject, depth + 1);
        }
    }
}