using UnityEngine;
using UnityEngine.SceneManagement;
using System.Collections;
using System;
public class MainMenuController : MonoBehaviour
{
    [SerializeField] private GameObject aboutUsPanel;
    [SerializeField] private GameObject buttons;
    [SerializeField] private GameObject aboutUsButton;

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
        if (aboutUsPanel.activeSelf)
        {
            //setTimeout to deactivate the about us panel after 0.1 seconds to allow the buttons to slide back first
            StartCoroutine(SetActiveAboutUsPanelAfterDelay(0.1f, false));
            // move buttons back to x=0
            StartCoroutine(SlideButtonsRoutine(new Vector3(0, buttons.transform.localPosition.y, 0)));
        }
        else
        {
            //setTimeout to activate the about us panel after 0.4 seconds to allow the buttons to slide away first
            StartCoroutine(SetActiveAboutUsPanelAfterDelay(0.4f, true));
            // move buttons away to x=-270 to make room for about us panel
            StartCoroutine(SlideButtonsRoutine(new Vector3(-270, buttons.transform.localPosition.y, 0)));
        }
        aboutUsButton.GetComponent<UnityEngine.UI.Button>().interactable = false;
        StartCoroutine(ReenableAboutUsButtonAfterDelay(animationDuration+0.02f));

    }
    private IEnumerator ReenableAboutUsButtonAfterDelay(float delay)
    {
        yield return new WaitForSeconds(delay);
        aboutUsButton.GetComponent<UnityEngine.UI.Button>().interactable = true;
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
    private IEnumerator SetActiveAboutUsPanelAfterDelay(float delay,bool set)
    {
        yield return new WaitForSeconds(delay);
        aboutUsPanel.SetActive(set);
    }
}
