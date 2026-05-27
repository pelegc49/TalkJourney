using UnityEngine;
using UnityEngine.SceneManagement;
using System.Collections;
using System;
public class MainMenuController : MonoBehaviour
{
    [SerializeField] private GameObject aboutUsPanel;
    [SerializeField] private GameObject buttons;

    private float animationDuration = 0.5f;
    public void onStartPressed()
    {
        SceneManager.LoadScene("City");
        //SceneManager.UnloadSceneAsync("MainMenu");
    }

    public void onExitPressed()
    {
        Debug.Log("Exiting game...");
        Application.Quit();
    }

    public void onAboutUsPressed()
    {
        //setTimeout to activate the about us panel after 0.4 seconds to allow the buttons to slide away first
        StartCoroutine(ActivateAboutUsPanelAfterDelay(0.4f));
        // move buttons away to x=-270 to make room for about us panel
        StartCoroutine(SlideButtonsRoutine(new Vector3(-270, 45, 0)));
    }
    private IEnumerator SlideButtonsRoutine(Vector3 targetPosition)
    {
        Vector3 startPosition = buttons.transform.localPosition;
        float elapsedTime = 0f;

        while (elapsedTime < animationDuration)
        {
            // Linear interpolation normalizes the movement over the specified duration.
            buttons.transform.localPosition = Vector3.Lerp(startPosition, targetPosition, elapsedTime / animationDuration);
            elapsedTime += Time.deltaTime;

            // Yielding null pauses execution until the next frame.
            yield return null;
        }

        // Guarantees precision by snapping to the exact target coordinate upon completion,
        // mitigating floating point inaccuracies from the lerp.
        buttons.transform.localPosition = targetPosition;
    }
    private IEnumerator ActivateAboutUsPanelAfterDelay(float delay)
    {
        yield return new WaitForSeconds(delay);
        aboutUsPanel.SetActive(true);
    }
}
