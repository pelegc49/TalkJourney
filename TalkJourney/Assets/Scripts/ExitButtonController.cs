using UnityEngine;

public class ExitButtonController : MonoBehaviour
{
   public void onExitPressed()
    {
        Debug.Log("Exiting game...");
        Application.Quit();
    }
}
